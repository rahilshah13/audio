import os, json, pickle, jax, optax, random, time, glob, re, tempfile
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from dashboard import TrainingDashboard

class CALM(nn.Module):
    dim: int = 1024
    @nn.compact
    def __call__(self, x, return_attn: bool = False):
        x = nn.Dense(self.dim, name="down_proj_2")(nn.gelu(nn.Dense(2048, name="down_proj_1")(x)))        
        B, T, C = x.shape
        num_heads = 8
        head_dim = self.dim // num_heads
        q = nn.Dense(self.dim, name="query")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        k = nn.Dense(self.dim, name="key")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        v = nn.Dense(self.dim, name="value")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)        
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
        tril = jnp.tril(jnp.ones((T, T), dtype=bool))
        mask = tril[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        h = jnp.matmul(attn_weights, v).swapaxes(1, 2).reshape(B, T, C)
        h = nn.Dense(self.dim, name="attn_out")(h)
        h = nn.LayerNorm(name="ln_1")(h + x)
        ff = nn.Dense(self.dim, name="ff_2")(nn.gelu(nn.Dense(self.dim * 2, name="ff_1")(h)))
        out = nn.Dense(88200, name="up_proj_2")(nn.gelu(nn.Dense(2048, name="up_proj_1")(nn.LayerNorm(name="ln_2")(h + ff))))
        if return_attn: return out, attn_weights
        return out

# --- METADATA ---
def sharded_memmap_loader(batch_size=4, seq_len=20, samples_per_sec=44100):
    meta_path = "data/audio_vault.meta.jsonl"
    if not os.path.exists(meta_path):
        os.makedirs("data", exist_ok=True)
        with open(meta_path, "w") as f:
            f.write(json.dumps({"shard": "mock.bin", "num_samples": 8820000, "sample_rate": 44100, "offset_bytes": 0, "url": "mock_source"}) + "\n")
        with open("data/mock.bin", "wb") as f:
            f.write(np.random.randn(8820000, 2).astype(np.float32).tobytes())

    with open(meta_path, "r") as f: metadata = [json.loads(l) for l in f if l.strip()]
    mmap_pool = {}
    while True:
        batch, batch_urls, start_seconds = [], set(), []
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            bin_path = os.path.join("data", entry["shard"])
            if entry["shard"] not in mmap_pool:
                mmap_pool[entry["shard"]] = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 2)
            
            total_sec = entry["num_samples"] / entry["sample_rate"]
            latents = []
            chosen_start_offset = random.uniform(0, total_sec - seq_len)
            if len(start_seconds) == 0:
                start_seconds.append(int(chosen_start_offset))
                
            for i in range(seq_len):
                s_idx = (entry["offset_bytes"] // 8) + int((chosen_start_offset + i) * entry["sample_rate"])
                chunk = mmap_pool[entry["shard"]][s_idx : s_idx + samples_per_sec]
                if len(chunk) < samples_per_sec:
                    padded = np.zeros((samples_per_sec, 2), dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                latents.append(chunk.flatten())
            batch.append(jnp.stack(latents))
            batch_urls.add(entry["url"])
        yield jnp.stack(batch), batch_urls, start_seconds[0]

# ------------------------------ 
model, key = CALM(), jax.random.PRNGKey(42)
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/checkpoint_run.pickle"
params = model.init(key, jnp.zeros((1, 20, 88200)))['params']

if os.path.exists(checkpoint_path):
    print(f"\n[SYSTEM] Found existing parameter checkpoint file at: {checkpoint_path}")
    print("[SYSTEM] Synchronizing weights and resuming previous execution sequence...")
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
else:
    print("\n[SYSTEM] No previous checkpoints discovered. Commencing clean initialization parameters...")

# Store initialization parameter state for NTK evaluation reference tracking
initial_ntk_weights = np.array(params['up_proj_2']['kernel'])

tx = optax.adam(2e-4)
opt_state = tx.init(params)

def loss_fn(params, x, key, noise_scale):
    noised = x + jax.random.normal(jax.random.split(key)[0], x.shape) * noise_scale
    return jnp.mean(jnp.square(model.apply({'params': params}, noised[:, :-1, :]) - x[:, 1:, :]))

@jax.jit
def train_step(params, opt_state, batch, key, noise_scale):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key, noise_scale)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

TOTAL_STEPS = 50000
global_seen_urls = set()
board = TrainingDashboard(total_steps=TOTAL_STEPS)
loader = sharded_memmap_loader(batch_size=4, seq_len=20)

# Track min/max loss boundaries for 0-1 scale calculation
min_loss = float('inf')
max_loss = float('-inf')

print("\nExecuting Training Iterations. Dashboard running on optimized UI pass loops...\n")

try:
    for step in range(1, TOTAL_STEPS + 1):
        batch_data, step_urls, window_start_sec = next(loader)
        global_seen_urls.update(step_urls)
        step_noise_scale = float(random.uniform(0.01, 0.08))
        key, step_key = jax.random.split(key)        
        params, opt_state, loss = train_step(params, opt_state, batch_data, step_key, step_noise_scale)        
        loss_val = float(loss)
        
        # Track historical boundaries dynamically
        if loss_val < min_loss: min_loss = loss_val
        if loss_val > max_loss: max_loss = loss_val        
        denom = max_loss - min_loss
        scaled_loss = (loss_val - min_loss) / denom if denom > 1e-8 else 0.5
        
        if step % 10 == 0 or step == 1:
            print(f"\rProgress: {(step / TOTAL_STEPS) * 100:6.2f}% | Step {step}/{TOTAL_STEPS} | Abs Loss: {loss_val:.4f} | Scaled Loss: {scaled_loss:.4f} | Noise Scale: {step_noise_scale:.3f}", end="", flush=True)            
            _, weights_tensor = model.apply({'params': params}, batch_data, return_attn=True)            
            current_sample_title = list(step_urls)[0] if step_urls else "Unknown Source"            
            
            # Extract current parameter state to evaluate the empirical NTK delta tracking
            current_ntk_weights = np.array(params['up_proj_2']['kernel'])
            
            board.update(
                step=step, 
                loss_val=loss_val, 
                noise_scale=step_noise_scale, 
                seen_count=len(global_seen_urls), 
                weights_tensor=weights_tensor, 
                raw_visual_waveform=np.array(batch_data[0]), 
                sample_title=current_sample_title, 
                window_start_sec=window_start_sec,
                current_ntk=current_ntk_weights,
                initial_ntk=initial_ntk_weights
            )
            
        if step % 1000 == 0:
            with open(checkpoint_path, "wb") as f: 
                pickle.dump(params, f)
                
except KeyboardInterrupt:
    print("\nExecution safely interrupted by user request.")
finally:
    with open(checkpoint_path, "wb") as f: 
        pickle.dump(params, f)
    print("\nCheckpoints successfully synchronized with persistent volume structures.")