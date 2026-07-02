import os, sys, glob, pickle, jax, json, random
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav
from flax import linen as nn

class CALM(nn.Module):
    dim: int = 128
    @nn.compact
    def __call__(self, x):
        mask = nn.make_causal_mask(jnp.ones(x.shape[1]))
        h = nn.SelfAttention(num_heads=4, qkv_features=self.dim)(x, mask=mask)
        h = nn.LayerNorm()(h + x)
        h = nn.Dense(self.dim * 2)(h)
        h = nn.gelu(h)
        return nn.Dense(self.dim)(h)

try:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
except ValueError:
    duration = 10.0

SR, CHUNK_SEC, SEQ_LEN, DIM = 44100, 0.2, 64, 128
CHUNKS_NEEDED = int(np.ceil(duration / CHUNK_SEC))
SAMPLES_PER_CHUNK = int(SR * CHUNK_SEC)

cps = sorted(glob.glob("checkpoints/checkpoint_run_*.pickle"), key=os.path.getmtime)
if not cps: raise FileNotFoundError("No checkpoints found.")
with open(cps[-1], "rb") as f: params = pickle.load(f)

# --- Real Audio Priming Logic ---
meta_path = "data/audio_vault.meta.jsonl"
if not os.path.exists(meta_path): raise FileNotFoundError("Missing data ledger.")
with open(meta_path, "r") as f: metadata = [json.loads(l) for l in f if l.strip()]

seed_latents = []
while len(seed_latents) < SEQ_LEN:
    entry = random.choice(metadata)
    bin_path = os.path.join("data", entry["shard"])
    if not os.path.exists(bin_path): continue
    
    mmap_data = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 2)
    offset_samples = entry["offset_bytes"] // 8
    total_samples, sr = entry["num_samples"], entry["sample_rate"]
    
    if (total_samples / sr) <= (SEQ_LEN * CHUNK_SEC): continue
    start_sec = random.uniform(0, (total_samples / sr) - (SEQ_LEN * CHUNK_SEC))
    
    for i in range(SEQ_LEN):
        t = start_sec + (i * CHUNK_SEC)
        s_idx = offset_samples + int(t * sr)
        chunk = mmap_data[s_idx : s_idx + int(CHUNK_SEC * sr)] / 32768.0
        if len(chunk) < int(CHUNK_SEC * sr):
            chunk = np.zeros((int(CHUNK_SEC * sr), 2), dtype=np.float32)
            
        if len(chunk) >= 64:
            indices = np.linspace(0, len(chunk) - 1, 64).astype(np.int32)
            latent_vector = chunk[indices].flatten()
        else:
            latent_vector = np.zeros((128,), dtype=np.float32)
        seed_latents.append(latent_vector)

context = jnp.expand_dims(jnp.array(seed_latents), axis=0)
# --------------------------------

model, generated = CALM(), []

for _ in range(CHUNKS_NEEDED):
    out = model.apply({'params': params}, context)
    nxt = out[:, -1, :]
    generated.append(np.array(nxt[0]))
    context = jnp.concatenate([context[:, 1:, :], jnp.expand_dims(nxt, axis=1)], axis=1)

audio_chunks = []
for latent in generated:
    pairs = latent.reshape(64, 2)
    xp = np.linspace(0, 1, 64)
    xnew = np.linspace(0, 1, SAMPLES_PER_CHUNK)
    l_ch = np.interp(xnew, xp, pairs[:, 0])
    r_ch = np.interp(xnew, xp, pairs[:, 1])
    audio_chunks.append(np.column_stack([l_ch, r_ch]))

waveform = np.vstack(audio_chunks)[:int(duration * SR)] * 32768.0
audio_out = np.clip(waveform, -32768, 32767).astype(np.int16)
wav.write(f"generated_output_{int(duration)}s.wav", SR, audio_out)