import os, sys, glob, pickle, jax, json, random, re
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav

try:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
except ValueError:
    duration = 10.0

SR = 44100
SEQ_LEN = 20 
STEPS_NEEDED = int(np.ceil(duration))

cps = sorted(glob.glob("checkpoints/checkpoint_run.pickle"), key=os.path.getmtime)
if not cps: 
    raise FileNotFoundError("No checkpoints found. Please run the training step first.")
print(f"Loading weights from {cps[-1]}...")
with open(cps[-1], "rb") as f: 
    params = pickle.load(f)

jaxpr_path = "checkpoints/model_jaxpr.pickle"
if not os.path.exists(jaxpr_path):
    raise FileNotFoundError(f"Missing JAXPR file at {jaxpr_path}. Ensure it was emitted by model.py.")
print(f"Loading JAXPR from {jaxpr_path}...")
with open(jaxpr_path, "rb") as f:
    closed_jaxpr = pickle.load(f)

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
    
    mmap_data = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 4)
    offset_samples = entry["offset_bytes"] // 16
    total_samples, sr = entry["num_samples"], entry["sample_rate"]
    
    if (total_samples / sr) <= SEQ_LEN: 
        continue
    start_sec = random.uniform(0, (total_samples / sr) - SEQ_LEN)
    
    seed_latents = []
    for i in range(SEQ_LEN):
        s_idx = offset_samples + int((start_sec + i) * sr)
        chunk = mmap_data[s_idx : s_idx + sr]
        
        if len(chunk) < sr:
            padded = np.zeros((sr, 4), dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
            
        if sr != SR:
            xp = np.linspace(0, 1, len(chunk))
            xnew = np.linspace(0, 1, SR)
            v_l = np.interp(xnew, xp, chunk[:, 0])
            v_r = np.interp(xnew, xp, chunk[:, 1])
            i_l = np.interp(xnew, xp, chunk[:, 2])
            i_r = np.interp(xnew, xp, chunk[:, 3])
            chunk = np.column_stack([v_l, v_r, i_l, i_r])
        else:
            if len(chunk) != SR:
                padded = np.zeros((SR, 4), dtype=np.float32)
                min_len = min(len(chunk), SR)
                padded[:min_len] = chunk[:min_len]
                chunk = padded
                
        seed_latents.append(chunk.flatten() / 32768.0)

context = jnp.expand_dims(jnp.array(seed_latents), axis=0)

@jax.jit
def run_jaxpr(p, x):
    return jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, p, x)[0]

generated = []
print(f"Starting autoregressive inference for {duration} seconds using JAXPR...")

for step in range(STEPS_NEEDED):
    out = run_jaxpr(params, context)
    nxt = out[:, -1, :] 
    generated.append(np.array(nxt[0]))    
    context = jnp.concatenate([context[:, 1:, :], jnp.expand_dims(nxt, axis=1)], axis=1)
    print(f"Generated second {step + 1}/{STEPS_NEEDED}")

waveform = np.concatenate(generated).reshape(-1, 4) * 32768.0
waveform = waveform[:int(duration * SR)] 

vocals = waveform[:, :2]
instrumentals = waveform[:, 2:]

vocals_out = np.clip(vocals, -32768, 32767).astype(np.int16)
inst_out = np.clip(instrumentals, -32768, 32767).astype(np.int16)

out_vocals_name = f"generated_vocals_{int(duration)}s.wav"
out_inst_name = f"generated_instrumentals_{int(duration)}s.wav"

wav.write(out_vocals_name, SR, vocals_out)
wav.write(out_inst_name, SR, inst_out)

print(f"Saved generated vocals to: {out_vocals_name}")
print(f"Saved generated instrumentals to: {out_inst_name}")
