import os, json, pickle, jax, optax, random, time, glob, re
import jax.numpy as jnp
import numpy as np
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

def sharded_memmap_loader(batch_size=4, seq_len=64, chunk_sec=0.2):
    meta_path = "data/audio_vault.meta.jsonl"
    if not os.path.exists(meta_path):
        print("Data ledger missing.")
        return

    with open(meta_path, "r") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    
    mmap_pool = {}

    while True:
        batch = []
        batch_urls = set()
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            shard_file = entry["shard"]
            bin_path = os.path.join("data", shard_file)
            
            if not os.path.exists(bin_path): continue
            
            if shard_file not in mmap_pool:
                mmap_pool[shard_file] = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 2)
                
            mmap_data = mmap_pool[shard_file]
            offset_samples = entry["offset_bytes"] // (4 * 2)
            total_samples, sr = entry["num_samples"], entry["sample_rate"]
            
            total_sec = total_samples / sr
            if total_sec <= (seq_len * chunk_sec): continue
            
            start_sec = random.uniform(0, total_sec - (seq_len * chunk_sec))
            latents = []
            for i in range(seq_len):
                t = start_sec + (i * chunk_sec)
                s_idx = offset_samples + int(t * sr)
                e_idx = s_idx + int(chunk_sec * sr)
                
                chunk = mmap_data[s_idx:e_idx]
                if len(chunk) < int(chunk_sec * sr):
                    chunk = np.zeros((int(chunk_sec * sr), 2), dtype=np.float32)
                
                chunk = chunk / 32768.0 
                if len(chunk) >= 64:
                    indices = np.linspace(0, len(chunk) - 1, 64).astype(np.int32)
                    latent_vector = chunk[indices].flatten() 
                else:
                    latent_vector = np.zeros((128,), dtype=np.float32)

                latents.append(latent_vector)
            
            batch.append(jnp.stack(latents))
            batch_urls.add(entry["url"])
        yield jnp.stack(batch), batch_urls

def loss_fn(params, x, key):
    k1, k2 = jax.random.split(key)
    noise = jax.random.normal(k1, x.shape) * 0.1
    noised_inputs = x + noise
    predictions = model.apply({'params': params}, noised_inputs[:, :-1, :])
    targets = x[:, 1:, :]
    return jnp.mean(jnp.square(predictions - targets))

model = CALM()
key = jax.random.PRNGKey(42)
x_init = jnp.zeros((1, 64, 128))

os.makedirs("checkpoints", exist_ok=True)
checkpoint_files = glob.glob("checkpoints/checkpoint_run_*.pickle")

if checkpoint_files:
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]
    print(f"Found existing progress! Loading weights from {latest_checkpoint}...")
    with open(latest_checkpoint, "rb") as f:
        params = pickle.load(f)
    run_id = re.search(r"checkpoint_(run_\d+)", latest_checkpoint).group(1)
    metadata_path = f"checkpoints/checkpoint_{run_id}.txt"
    global_seen_urls = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as tf:
            global_seen_urls = set(tf.read().splitlines())
else:
    print("No previous checkpoints found. Initializing a fresh model...")
    params = model.init(key, x_init)['params']
    run_id = f"run_{int(time.time())}"
    global_seen_urls = set()

tx = optax.adam(learning_rate=3e-4)
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

loader = sharded_memmap_loader()
checkpoint_path = f"checkpoints/checkpoint_{run_id}.pickle"
metadata_path = f"checkpoints/checkpoint_{run_id}.txt"

TOTAL_STEPS = 100000
print(f"Starting training loop. Checkpoint ID: {run_id}.")

try:
    for step in range(1, TOTAL_STEPS + 1):
        batch_data, step_urls = next(loader)
        global_seen_urls.update(step_urls)
        
        key, step_key = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, batch_data, step_key)
        
        if step % 10 == 0 or step == 1:
            progress = (step / TOTAL_STEPS) * 100
            print(f"\rProgress: {progress:6.2f}% | Step {step}/{TOTAL_STEPS} | Loss: {loss:.6f}", end="", flush=True)
        
        if step % 1000 == 0:
            print(f"\n[Checkpoint] Saving step {step} weights to disk...")
            with open(checkpoint_path, "wb") as f:
                pickle.dump(params, f)
            with open(metadata_path, "w") as tf:
                tf.write("\n".join(global_seen_urls) + "\n")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user. Saving final run state...")
finally:
    with open(checkpoint_path, "wb") as f:
        pickle.dump(params, f)
    with open(metadata_path, "w") as tf:
        tf.write("\n".join(global_seen_urls) + "\n")
    print(f"Saved master checkpoint: {checkpoint_path}")