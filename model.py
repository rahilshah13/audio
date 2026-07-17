import os, json, pickle, jax, optax, random, time, numpy as np
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from dashboard import TrainingDashboard
from meta import get_calm_params_from_ntk_trajectory
import fcntl 

class CALM(nn.Module):
    dim: int = 4096 
    @nn.compact
    def __call__(self, x, return_attn: bool = False):
        x = nn.Dense(self.dim, name="down_proj_2")(nn.gelu(nn.Dense(8192, name="down_proj_1")(x)))        
        B, T, C = x.shape
        num_heads = 16
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
        h = nn.LayerNorm(name="ln_1")(h + x)
        ff = nn.Dense(self.dim, name="ff_2")(nn.gelu(nn.Dense(self.dim * 4, name="ff_1")(h)))
        out = nn.Dense(176400, name="up_proj_2")(nn.gelu(nn.Dense(8192, name="up_proj_1")(nn.LayerNorm(name="ln_2")(h + ff))))
        if return_attn: return out, attn_weights
        return out

def normalize_loss(loss, scale=0.5):
    return 1.0 - np.exp(-loss / scale)

STATE_FILE = "data/global_state.json"
GRADIENT_STORE = "checkpoints/accumulated_gradients.pickle"

def read_global_state():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(STATE_FILE):
        return {"processed_windows": []} 
    try:
        with open(STATE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except Exception:
        return {"processed_windows": []}

def register_global_window(window_str):
    with open(STATE_FILE, "r+" if os.path.exists(STATE_FILE) else "w+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = json.load(f)
        except Exception:
            data = {"processed_windows": []}
        data["processed_windows"].append(window_str)
        f.seek(0)
        f.truncate()
        json.dump(data, f)
        fcntl.flock(f, fcntl.LOCK_UN)

def push_and_pull_gradients(local_grads, accumulation_steps=1000):
    grad_store_path = "data/shared_gradients.pickle"
    params_store_path = "checkpoints/checkpoint_run.pickle"
    
    with open(grad_store_path, "a+b" if os.path.exists(grad_store_path) else "w+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            shared_data = pickle.load(f)
        except Exception:
            shared_data = {"accumulated_grads": None, "count": 0}
        
        if shared_data["accumulated_grads"] is None:
            shared_data["accumulated_grads"] = local_grads
        else:
            shared_data["accumulated_grads"] = jax.tree_util.tree_map(
                lambda x, y: x + y, shared_data["accumulated_grads"], local_grads
            )
        shared_data["count"] += 1
        
        apply_global_update = False
        if shared_data["count"] >= accumulation_steps:
            apply_global_update = True
            shared_grads = jax.tree_util.tree_map(lambda x: x / shared_data["count"], shared_data["accumulated_grads"])
            shared_data = {"accumulated_grads": None, "count": 0} 
            
        f.seek(0)
        f.truncate()
        pickle.dump(shared_data, f)
        fcntl.flock(f, fcntl.LOCK_UN)
        
    if apply_global_update:
        with open(params_store_path, "r+b") as pf:
            fcntl.flock(pf, fcntl.LOCK_EX)
            global_params = pickle.load(pf)
            
            tx = optax.adam(2e-4)
            opt_state_path = "checkpoints/opt_state.pickle"
            if os.path.exists(opt_state_path):
                with open(opt_state_path, "rb") as os_f: opt_state = pickle.load(os_f)
            else:
                opt_state = tx.init(global_params)
                
            updates, new_opt_state = tx.update(shared_grads, opt_state, global_params)
            global_params = optax.apply_updates(global_params, updates)
            
            pf.seek(0)
            pf.truncate()
            pickle.dump(global_params, pf)
            
            with open(opt_state_path, "wb") as os_f:
                pickle.dump(new_opt_state, os_f)
                
            fcntl.flock(pf, fcntl.LOCK_UN)
            return global_params, True
            
    with open(params_store_path, "rb") as pf:
        fcntl.flock(pf, fcntl.LOCK_SH)
        global_params = pickle.load(pf)
        fcntl.flock(pf, fcntl.LOCK_UN)
    return global_params, False

def daemon_memmap_loader(batch_size, seq_len=10, samples_per_sec=44100):
    meta_path = "data/audio_vault.meta.jsonl"
    mmap_pool = {}
    
    while True:
        if not os.path.exists(meta_path):
            time.sleep(2)
            continue
        with open(meta_path, "r") as f:
            metadata = [json.loads(l) for l in f if l.strip()]
        
        if not metadata:
            time.sleep(2)
            continue
            
        batch, batch_urls = [], set()
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            shard_path = os.path.join("data", entry["shard"])
            
            if not os.path.exists(shard_path):
                continue
                
            curr_size = os.path.getsize(shard_path)
            mmap_pool[entry["shard"]] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1, 4)
            
            total_samples = curr_size // 16 
            max_start_sec = (total_samples / entry["sample_rate"]) - seq_len
            if max_start_sec <= 0:
                continue
                
            chosen_start = random.uniform(0, max_start_sec)
            start_idx = int(chosen_start * entry["sample_rate"])
            end_idx = start_idx + (seq_len * samples_per_sec)
            
            window_id = f"{entry['shard']}:{start_idx}:{end_idx}"
            g_state = read_global_state()
            if window_id in g_state["processed_windows"]:
                continue 
                
            register_global_window(window_id)
            
            latents = []
            for i in range(seq_len):
                s_idx = (entry["offset_bytes"] // 16) + start_idx + (i * samples_per_sec)
                chunk = mmap_pool[entry["shard"]][s_idx : s_idx + samples_per_sec]
                if len(chunk) < samples_per_sec:
                    padded = np.zeros((samples_per_sec, 4), dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                latents.append(chunk.flatten())
                
            batch.append(jnp.stack(latents))
            batch_urls.add(entry["url"])
            
        yield jnp.stack(batch), batch_urls

def make_ntk_fn(model):
    def model_forward_flat(params, x):
        return model.apply({'params': params}, x).flatten()
    @jax.jit
    def compute_ntk(params, x):
        jac = jax.jacobian(model_forward_flat, argnums=0)(params, x)
        jac_flat = jnp.concatenate([jnp.reshape(j, (j.shape[0], -1)) for j in jax.tree_util.tree_leaves(jac)], axis=-1)
        return jnp.matmul(jac_flat, jac_flat.T)
    return compute_ntk

if __name__ == "__main__":
    model = CALM()
    key = jax.random.PRNGKey(42)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("ntk_logs", exist_ok=True)
    checkpoint_path = "checkpoints/checkpoint_run.pickle"
    jaxpr_path = "checkpoints/model_jaxpr.pickle"
    
    if not os.path.exists(checkpoint_path):
        initial_params = model.init(key, jnp.zeros((1, 10, 176400)))['params']
        with open(checkpoint_path, "wb") as f:
            pickle.dump(initial_params, f)

    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)

    infer_dummy = jnp.zeros((1, 20, 176400))
    closed_jaxpr = jax.make_jaxpr(lambda p, x: model.apply({'params': p}, x))(params, infer_dummy)
    with open(jaxpr_path, "wb") as f: pickle.dump(closed_jaxpr, f)

    ntk_calculator = make_ntk_fn(model)
    MICRO_BATCH_SIZE = 1
    loader = daemon_memmap_loader(batch_size=MICRO_BATCH_SIZE)

    @partial(jax.jit, static_argnames=['noise_scale'])
    def micro_step(params, batch, key, noise_scale):
        noised = batch + jax.random.normal(jax.random.split(key)[0], batch.shape) * noise_scale
        preds = model.apply({'params': params}, noised[:, :-1, :])
        loss = jnp.mean(jnp.square(preds - batch[:, 1:, :]))
        return loss, jax.grad(lambda p: jnp.mean(jnp.square(model.apply({'params': p}, noised[:, :-1, :]) - batch[:, 1:, :])))(params)

    board = TrainingDashboard(total_steps=100000)
    step = 1
    
    print("Distributed model worker daemon initialized and listening for streaming shards...")
    while True:
        try:
            b_data, b_urls = next(loader)
            loss, grads = micro_step(params, b_data, key, 0.05)
            
            params, global_updated = push_and_pull_gradients(grads, accumulation_steps=100)
            
            if global_updated:
                print(f"[Step {step}] Global Parameter update completed. Loss: {float(loss):.5f}")
                if step % 100 == 0:
                    np.save(f"ntk_logs/ntk_step_{step}.npy", np.array(ntk_calculator(params, jax.random.normal(key, (1, 1, 176400)))))
                    closed_jaxpr = jax.make_jaxpr(lambda p, x: model.apply({'params': p}, x))(params, infer_dummy)
                    with open(jaxpr_path, "wb") as f: pickle.dump(closed_jaxpr, f)
                step += 1
                
        except Exception as e:
            print(f"Error in training step: {e}")
            time.sleep(1)
