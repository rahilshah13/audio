import os, sys, glob, pickle, jax, json, random, re
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav

from model import CALM

try:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
except ValueError:
    duration = 10.0

SR = 44100
SEQ_LEN = 20 # 20-second context window
STEPS_NEEDED = int(np.ceil(duration))

cps = sorted(glob.glob("checkpoints/checkpoint_run.pickle"), key=os.path.getmtime)
if not cps: 
    raise FileNotFoundError("No checkpoints found. Please run the training step first.")
print(f"Loading weights from {cps[-1]}...")
with open(cps[-1], "rb") as f: 
    params = pickle.load(f)

inferred_dim = params['down_proj_2']['kernel'].shape[-1]
print(f"Detected model hidden dimension from checkpoint: {inferred_dim}")

meta_path = "data/audio_vault.meta.jsonl"
if not os.path.exists(meta_path): 
    raise FileNotFoundError("Missing data ledger.")
with open(meta_path, "r") as f: 
    metadata = [json.loads(l) for l in f if l.strip()]

seed_latents = []
while len(seed_latents) < SEQ_LEN:
    entry = random.choice(metadata)
    bin_path = os.path.join("data", entry["shard"])
    if not os.path.exists(bin_path): 
        continue
    
    mmap_data = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 2)
    offset_samples = entry["offset_bytes"] // 8
    total_samples, sr = entry["num_samples"], entry["sample_rate"]
    
    if (total_samples / sr) <= SEQ_LEN: 
        continue
    start_sec = random.uniform(0, (total_samples / sr) - SEQ_LEN)
    
    seed_latents = []
    for i in range(SEQ_LEN):
        s_idx = offset_samples + int((start_sec + i) * sr)
        chunk = mmap_data[s_idx : s_idx + sr]
        
        if len(chunk) < sr:
            padded = np.zeros((sr, 2), dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
            
        if sr != SR:
            xp = np.linspace(0, 1, len(chunk))
            xnew = np.linspace(0, 1, SR)
            l_ch = np.interp(xnew, xp, chunk[:, 0])
            r_ch = np.interp(xnew, xp, chunk[:, 1])
            chunk = np.column_stack([l_ch, r_ch])
        else:
            if len(chunk) != SR:
                padded = np.zeros((SR, 2), dtype=np.float32)
                min_len = min(len(chunk), SR)
                padded[:min_len] = chunk[:min_len]
                chunk = padded
                
        seed_latents.append(chunk.flatten() / 32768.0)

context = jnp.expand_dims(jnp.array(seed_latents), axis=0)

model = CALM(dim=inferred_dim)
generated = []
print(f"Starting autoregressive inference for {duration} seconds...")

for step in range(STEPS_NEEDED):
    out = model.apply({'params': params}, context)
    nxt = out[:, -1, :] # Predict the final chronological step
    generated.append(np.array(nxt[0]))    
    context = jnp.concatenate([context[:, 1:, :], jnp.expand_dims(nxt, axis=1)], axis=1)
    print(f"Generated second {step + 1}/{STEPS_NEEDED}")

waveform = np.concatenate(generated).reshape(-1, 2) * 32768.0
waveform = waveform[:int(duration * SR)] 
audio_out = np.clip(waveform, -32768, 32767).astype(np.int16)

out_name = f"generated_output_{int(duration)}s.wav"
wav.write(out_name, SR, audio_out)
print(f"Saved generated sequence output to: {out_name}")