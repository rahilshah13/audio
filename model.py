import os, json, pickle, jax, optax, random, time, numpy as np
import jax.numpy as jnp
from functools import partial
from meta import get_meta_preconditioner
import fcntl 

STATE_FILE = "data/global_state.json"

def gpt_forward(params, x, scale, bpm, n_heads=16):
    # Initial projection
    x = jax.nn.gelu(x @ params['down_proj_1']) @ params['down_proj_2']
    B, T, C = x.shape
    
    # Feature Conditioning: Embed scale and project BPM
    s_emb = params['scale_emb'][scale]  # [B, C]
    b_emb = bpm[:, None] @ params['bpm_proj']  # [B, 1] @ [1, C] -> [B, C]
    cond = jnp.expand_dims(s_emb + b_emb, 1)  # [B, 1, C]
    
    # Inject condition into the sequence
    x = x + cond 
    
    head_dim = C // n_heads
    
    q = (x @ params['query']).reshape(B, T, n_heads, head_dim).swapaxes(1, 2)
    k = (x @ params['key']).reshape(B, T, n_heads, head_dim).swapaxes(1, 2)
    v = (x @ params['value']).reshape(B, T, n_heads, head_dim).swapaxes(1, 2)
    
    scores = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
    scores = jnp.where(mask, scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1) @ v
    
    h = attn.swapaxes(1, 2).reshape(B, T, C)
    h = ((h + x) - jnp.mean(h + x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(h + x, axis=-1, keepdims=True) + 1e-5)
    
    ff = jax.nn.gelu(h @ params['ff_1']) @ params['ff_2']
    h_norm = ((h + ff) - jnp.mean(h + ff, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(h + ff, axis=-1, keepdims=True) + 1e-5)
    
    return jax.nn.gelu(h_norm @ params['up_proj_1']) @ params['up_proj_2']

def init_params(key, dim=4096):
    keys = jax.random.split(key, 12)
    return {
        'down_proj_1': jax.random.normal(keys[0], (dim, 8192)),
        'down_proj_2': jax.random.normal(keys[1], (8192, dim)),
        'query': jax.random.normal(keys[2], (dim, dim)),
        'key': jax.random.normal(keys[3], (dim, dim)),
        'value': jax.random.normal(keys[4], (dim, dim)),
        'ff_1': jax.random.normal(keys[5], (dim, dim * 4)),
        'ff_2': jax.random.normal(keys[6], (dim * 4, dim)),
        'up_proj_1': jax.random.normal(keys[7], (dim, 8192)),
        'up_proj_2': jax.random.normal(keys[8], (8192, 176400)),
        'scale_emb': jax.random.normal(keys[9], (128, dim)),  # Assumes scale is categorical (0-127)
        'bpm_proj': jax.random.normal(keys[10], (1, dim))     # Projects continuous float BPM
    }

def read_global_state():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(STATE_FILE): return {"processed_windows": []} 
    try:
        with open(STATE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except Exception: return {"processed_windows": []}

def register_global_window(window_str):
    with open(STATE_FILE, "r+" if os.path.exists(STATE_FILE) else "w+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = json.load(f)
        except Exception: data = {"processed_windows": []}
        data["processed_windows"].append(window_str)
        f.seek(0); f.truncate(); json.dump(data, f)
        fcntl.flock(f, fcntl.LOCK_UN)

def push_and_pull_gradients(local_grads, accumulation_steps=1000):
    grad_store_path = "data/shared_gradients.pickle"
    params_store_path = "checkpoints/checkpoint_run.pickle"
    with open(grad_store_path, "a+b" if os.path.exists(grad_store_path) else "w+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try: shared_data = pickle.load(f)
        except: shared_data = {"accumulated_grads": None, "count": 0}
        if shared_data["accumulated_grads"] is None: shared_data["accumulated_grads"] = local_grads
        else: shared_data["accumulated_grads"] = jax.tree_util.tree_map(lambda x, y: x + y, shared_data["accumulated_grads"], local_grads)
        shared_data["count"] += 1
        apply_global_update = shared_data["count"] >= accumulation_steps
        if apply_global_update:
            shared_grads = jax.tree_util.tree_map(lambda x: x / shared_data["count"], shared_data["accumulated_grads"])
            shared_data = {"accumulated_grads": None, "count": 0} 
        f.seek(0); f.truncate(); pickle.dump(shared_data, f)
        fcntl.flock(f, fcntl.LOCK_UN)
    if apply_global_update:
        with open(params_store_path, "r+b") as pf:
            fcntl.flock(pf, fcntl.LOCK_EX)
            params = pickle.load(pf)
            preconditioner = get_meta_preconditioner(shared_grads)
            if preconditioner:
                shared_grads = jax.tree_util.tree_map(lambda g, p: g * p, shared_grads, preconditioner)
            tx = optax.adam(2e-4)
            opt_state_path = "checkpoints/opt_state.pickle"
            opt_state = pickle.load(open(opt_state_path, "rb")) if os.path.exists(opt_state_path) else tx.init(params)
            updates, new_opt_state = tx.update(shared_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            pf.seek(0); pf.truncate(); pickle.dump(params, pf)
            pickle.dump(new_opt_state, open(opt_state_path, "wb"))
            fcntl.flock(pf, fcntl.LOCK_UN)
            return params, True
    with open(params_store_path, "rb") as pf:
        fcntl.flock(pf, fcntl.LOCK_SH)
        params = pickle.load(pf)
        fcntl.flock(pf, fcntl.LOCK_UN)
    return params, False

def daemon_memmap_loader(batch_size, seq_len=10, samples_per_sec=44100):
    meta_path = "data/audio_vault.meta.jsonl"
    mmap_pool = {}
    while True:
        if not os.path.exists(meta_path): time.sleep(2); continue
        with open(meta_path, "r") as f: metadata = [json.loads(l) for l in f if l.strip()]
        if not metadata: time.sleep(2); continue
        
        batch, batch_scales, batch_bpms = [], [], []
        
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            shard_path = os.path.join("data", entry["shard"])
            if not os.path.exists(shard_path): continue
            mmap_pool[entry["shard"]] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1, 4)
            start_idx = int(random.uniform(0, (os.path.getsize(shard_path)//16 / entry["sample_rate"]) - seq_len) * entry["sample_rate"])
            window_id = f"{entry['shard']}:{start_idx}"
            
            if window_id in read_global_state()["processed_windows"]: continue 
            register_global_window(window_id)
            
            latents = [mmap_pool[entry["shard"]][(entry["offset_bytes"]//16)+start_idx+(i*samples_per_sec):(entry["offset_bytes"]//16)+start_idx+(i+1)*samples_per_sec].flatten() for i in range(seq_len)]
            
            batch.append(jnp.stack(latents))
            # Safely extract scale and bpm, defaulting if missing
            batch_scales.append(int(entry.get("scale", 0)))
            batch_bpms.append(float(entry.get("bpm", 120.0)))
            
        yield jnp.stack(batch), jnp.array(batch_scales, dtype=jnp.int32), jnp.array(batch_bpms, dtype=jnp.float32)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("ntk_logs", exist_ok=True)
    if not os.path.exists("checkpoints/checkpoint_run.pickle"):
        with open("checkpoints/checkpoint_run.pickle", "wb") as f: pickle.dump(init_params(key), f)
    with open("checkpoints/checkpoint_run.pickle", "rb") as f: params = pickle.load(f)
    
    loader = daemon_memmap_loader(batch_size=1)
    
    @partial(jax.jit, static_argnames=['noise_scale'])
    def micro_step(params, batch, scales, bpms, key, noise_scale):
        noised = batch + jax.random.normal(jax.random.split(key)[0], batch.shape) * noise_scale
        loss_fn = lambda p: jnp.mean(jnp.square(gpt_forward(p, noised[:, :-1, :], scales, bpms) - batch[:, 1:, :]))
        return loss_fn(params), jax.grad(loss_fn)(params)
        
    step = 1
    while True:
        try:
            b_data, b_scales, b_bpms = next(loader)
            loss, grads = micro_step(params, b_data, b_scales, b_bpms, key, 0.05)
            params, global_updated = push_and_pull_gradients(grads, accumulation_steps=100)
            if global_updated:
                print(f"[Step {step}] Update. Loss: {float(loss):.5f}")
                step += 1
        except Exception as e: 
            print(e)
            time.sleep(1)
