"""
SUMMARY:
Inference script aligned with the optimized CALM architecture.
Generates high-quality 44.1kHz stereo audio by autoregressively predicting 
next-second wave sequences using a 20-second historical context.
"""

import os, sys, glob, pickle, jax, json, random, re
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav
from flax import linen as nn

# --- CONFORMANT CORE MODEL SPECIFICATION ---
class CALM(nn.Module):
    dim: int = 1024 # Must exactly match the optimized model.py
    
    @nn.compact
    def __call__(self, x, return_attn: bool = False):
        # Down-projection blocks
        x = nn.Dense(self.dim, name="down_proj_2")(nn.gelu(nn.Dense(2048, name="down_proj_1")(x)))
        
        # Explicit Multi-Head Self-Attention matching training setup
        B, T, C = x.shape
        num_heads = 8
        head_dim = self.dim // num_heads
        
        q = nn.Dense(self.dim, name="query")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        k = nn.Dense(self.dim, name="key")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        v = nn.Dense(self.dim, name="value")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        
        # Calculate scaled dot-product scores: shape (B, num_heads, T, T)
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
        
        # Explicit lower-triangular mask to guarantee strict autoregressive causality
        tril = jnp.tril(jnp.ones((T, T), dtype=bool))
        mask = tril[None, None, :, :]  # Shape: (1, 1, T, T) to cleanly broadcast
        
        scores = jnp.where(mask, scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        h = jnp.matmul(attn_weights, v).swapaxes(1, 2).reshape(B, T, C)
        h = nn.Dense(self.dim, name="attn_out")(h)
        
        h = nn.LayerNorm(name="ln_1")(h + x)
        ff = nn.Dense(self.dim, name="ff_2")(nn.gelu(nn.Dense(self.dim * 2, name="ff_1")(h)))
        out = nn.Dense(88200, name="up_proj_2")(nn.gelu(nn.Dense(2048, name="up_proj_1")(nn.LayerNorm(name="ln_2")(h + ff))))
        
        if return_attn:
            return out, attn_weights
        return out

try:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
except ValueError:
    duration = 10.0

SR = 44100
SEQ_LEN = 20 # Must match the 20-second context window of model.py
STEPS_NEEDED = int(np.ceil(duration))

# Synchronized path tracking matching the training environment's precise output parameters
cps = sorted(glob.glob("checkpoints/checkpoint_run.pickle"), key=os.path.getmtime)
if not cps: 
    raise FileNotFoundError("No checkpoints found. Please run the training step first.")
print(f"Loading weights from {cps[-1]}...")
with open(cps[-1], "rb") as f: 
    params = pickle.load(f)

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
        # Calculate offset relative to native sample rate
        s_idx = offset_samples + int((start_sec + i) * sr)
        chunk = mmap_data[s_idx : s_idx + sr]
        
        # Pad if short
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

model, generated = CALM(), []
print(f"Starting autoregressive inference for {duration} seconds...")

for step in range(STEPS_NEEDED):
    out = model.apply({'params': params}, context)
    nxt = out[:, -1, :] # Predict the final chronological step
    generated.append(np.array(nxt[0]))    
    context = jnp.concatenate([context[:, 1:, :], jnp.expand_dims(nxt, axis=1)], axis=1)
    print(f"Generated second {step + 1}/{STEPS_NEEDED}")

waveform = np.concatenate(generated).reshape(-1, 2) * 32768.0
waveform = waveform[:int(duration * SR)] # Trim exact length
audio_out = np.clip(waveform, -32768, 32767).astype(np.int16)

out_name = f"generated_output_{int(duration)}s.wav"
wav.write(out_name, SR, audio_out)
print(f"Saved generated sequence output to: {out_name}")