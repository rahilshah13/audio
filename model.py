import os, json, pickle, sys, jax, optax, random, time, numpy as np
import jax.numpy as jnp
from functools import partial
from meta import get_meta_preconditioner
import fcntl 

STATE_FILE = "data/global_state.json"

@jax.jit
def gpt_forward(params, x, scale, bpm, stems, target_dim=88200, n_heads=16):
    x = jax.nn.gelu(x @ params['down_proj_1']) @ params['down_proj_2']
    B, T, C = x.shape
    
    s_emb = params['scale_emb'][scale]
    b_emb = bpm[:, None] @ params['bpm_proj']
    st_emb = params['stem_emb'][stems]
    
    cond = jnp.expand_dims(s_emb + b_emb + st_emb, 1)
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

def init_params(key, dim=4096, target_dim=88200):
    keys = jax.random.split(key, 13)
    return {
        'down_proj_1': jax.random.normal(keys[0], (dim, 8192)),
        'down_proj_2': jax.random.normal(keys[1], (8192, dim)),
        'query': jax.random.normal(keys[2], (dim, dim)),
        'key': jax.random.normal(keys[3], (dim, dim)),
        'value': jax.random.normal(keys[4], (dim, dim)),
        'ff_1': jax.random.normal(keys[5], (dim, dim * 4)),
        'ff_2': jax.random.normal(keys[6], (dim * 4, dim)),
        'up_proj_1': jax.random.normal(keys[7], (dim, 8192)),
        'up_proj_2': jax.random.normal(keys[8], (8192, target_dim)), 
        'scale_emb': jax.random.normal(keys[9], (128, dim)),  
        'bpm_proj': jax.random.normal(keys[10], (1, dim)),    
        'stem_emb': jax.random.normal(keys[11], (2, dim))
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

@jax.jit
def quantize_and_merge_deltas(base_params, adapted_params, sparsity_threshold=0.01):
    def q_merge(base_p, adapt_p):
        delta = adapt_p - base_p
        magnitude = jnp.abs(delta)
        thresh = jnp.percentile(magnitude, 100.0 * (1.0 - sparsity_threshold))
        sparse_delta = jnp.where(magnitude >= thresh, delta, 0.0)
        quantized_delta = jnp.sign(sparse_delta) * jnp.round(jnp.abs(sparse_delta) * 10.0) / 10.0
        return base_p + quantized_delta

    return jax.tree_util.tree_map(q_merge, base_params, adapted_params)

def lie_group_perturbation_matching_update(params, shared_grads, loss_fn_builder, perturbation_scale=1e-4, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    
    flat_grads, treedef = jax.flatten_util.ravel_pytree(shared_grads)
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    
    dx = perturbation_scale * jax.random.normal(key, flat_params.shape)
    
    @jax.jit
    def grad_mapping(p_flat):
        p_tree = unflatten_fn(p_flat)
        _, g_tree = jax.value_and_grad(loss_fn_builder(p_tree))(p_tree)
        g_flat, _ = jax.flatten_util.ravel_pytree(g_tree)
        return g_flat

    _, dg = jax.jvp(grad_mapping, (flat_params,), (dx,))
    
    denom = jnp.dot(dx, dg) + 1e-8
    curvature_scalar = jnp.abs(jnp.dot(dx, flat_grads) / denom)
    clamped_scalar = jnp.clip(curvature_scalar, 1.0, 2.0)
    
    meta_precond = get_meta_preconditioner(shared_grads)
    if meta_precond is not None:
        flat_precond, _ = jax.flatten_util.ravel_pytree(meta_precond)
        lie_manifold_factors = jnp.clip(flat_precond * clamped_scalar, 1.0, 2.0)
    else:
        lie_manifold_factors = jnp.full_like(flat_grads, clamped_scalar)

    preconditioned_flat = flat_grads * lie_manifold_factors
    return treedef(preconditioned_flat)

def push_and_pull_gradients(local_grads, current_params, loss_fn_builder, accumulation_steps=100, lie_key=None):
    grad_store_path = "data/shared_gradients.pickle"
    params_store_path = "checkpoints/checkpoint_run.pickle"
    prev_params_store_path = "checkpoints/checkpoint_prev.pickle"
    
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
            
            if os.path.exists(params_store_path):
                with open(prev_params_store_path, "wb") as ppf:
                    pickle.dump(params, ppf)
            
            shared_grads = lie_group_perturbation_matching_update(params, shared_grads, loss_fn_builder, key=lie_key)
                
            tx = optax.adam(2e-4)
            opt_state_path = "checkpoints/opt_state.pickle"
            opt_state = pickle.load(open(opt_state_path, "rb")) if os.path.exists(opt_state_path) else tx.init(params)
            updates, new_opt_state = tx.update(shared_grads, opt_state, params)
            
            new_params = optax.apply_updates(params, updates)
            params = quantize_and_merge_deltas(params, new_params, sparsity_threshold=0.01)
            
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
        
        batch, batch_scales, batch_bpms, batch_stems = [], [], [], []
        
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            shard_path = os.path.join("data", entry["shard"])
            if not os.path.exists(shard_path): continue
            mmap_pool[entry["shard"]] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1, 2)
            bytes_per_frame = 8  
            start_idx = int(random.uniform(0, (os.path.getsize(shard_path)//bytes_per_frame / entry["sample_rate"]) - seq_len) * entry["sample_rate"])
            stem_type = int(entry.get("stem", 0)) 
            window_id = f"{entry['shard']}:{start_idx}:stem_{stem_type}"
            
            if window_id in read_global_state()["processed_windows"]: continue 
            register_global_window(window_id)
            
            offset_frames = entry["offset_bytes"] // bytes_per_frame
            latents = [
                mmap_pool[entry["shard"]][offset_frames + start_idx + (i * samples_per_sec) : offset_frames + start_idx + ((i + 1) * samples_per_sec)].flatten()
                for i in range(seq_len)
            ]
            
            batch.append(jnp.stack(latents))
            batch_scales.append(int(entry.get("scale", 0)))
            batch_bpms.append(float(entry.get("bpm", 120.0)))
            batch_stems.append(stem_type)
            
        yield jnp.stack(batch), jnp.array(batch_scales, dtype=jnp.int32), jnp.array(batch_bpms, dtype=jnp.float32), jnp.array(batch_stems, dtype=jnp.int32)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("ntk_logs", exist_ok=True)
    if not os.path.exists("checkpoints/checkpoint_run.pickle"):
        with open("checkpoints/checkpoint_run.pickle", "wb") as f: pickle.dump(init_params(key), f)
    with open("checkpoints/checkpoint_run.pickle", "rb") as f: params = pickle.load(f)
    
    loader = daemon_memmap_loader(batch_size=1)
    
    @partial(jax.jit, static_argnames=['noise_scale', 'num_diffusion_steps'])
    def train_step_until_zero(params, batch, scales, bpms, stems, key, noise_scale, num_diffusion_steps=10):
        k1, k2, k_loop = jax.random.split(key, 3)
        step_indices = jax.random.randint(k1, shape=(batch.shape[0],), minval=0, maxval=num_diffusion_steps)
        
        t = (step_indices.astype(jnp.float32) + 1.0) / float(num_diffusion_steps)
        t = t[:, None, None]
        alpha_t = jnp.cos(t * jnp.pi / 2.0)
        sigma_t = jnp.sin(t * jnp.pi / 2.0)
        
        noise = jax.random.normal(k2, batch.shape)
        noised = alpha_t * batch + sigma_t * noise * noise_scale
        
        noised_input = noised[:, :-1, :]
        target_output = batch[:, 1:, :]
        
        def loss_fn(p):
            pred_target = gpt_forward(p, noised_input, scales, bpms, stems)
            return jnp.mean(jnp.square(pred_target - target_output))

        def scan_body_fn(p, _):
            current_loss, grads = jax.value_and_grad(loss_fn)(p)
            next_p = jax.tree_util.tree_map(lambda param, g: param - 1e-4 * g, p, grads)
            return next_p, current_loss

        final_params, loss_history = jax.lax.scan(scan_body_fn, params, xs=None, length=50)
        
        final_loss, final_grads = jax.value_and_grad(loss_fn)(final_params)
        return final_loss, final_grads, k_loop

    sample_batch, sample_scales, sample_bpms, sample_stems = next(loader)
    _, sample_subk = jax.random.split(key)
    jaxpr_repr = jax.make_jaxpr(train_step_until_zero)(
        params, sample_batch, sample_scales, sample_bpms, sample_stems, sample_subk, 0.05
    )
    with open("checkpoints/train_step.jaxpr", "wb") as jpr_file:
        pickle.dump(jaxpr_repr, jpr_file)

    if "--export-jaxpr" in sys.argv:
        print("JAXPR exported to checkpoints/train_step.jaxpr. Exiting.")
        sys.exit(0)

    step = 1
    while True:
        try:
            b_data, b_scales, b_bpms, b_stems = next(loader)
            key, subkey = jax.random.split(key)
            loss, grads, key = train_step_until_zero(params, b_data, b_scales, b_bpms, b_stems, subkey, 0.05)
            
            loss_fn_builder = lambda current_p, bd=b_data, bs=b_scales, bp=b_bpms, st=b_stems: lambda p: jnp.mean(jnp.square(gpt_forward(p, bd[:, :-1, :], bs, bp, st) - bd[:, 1:, :]))
            
            key, lie_subkey = jax.random.split(key)
            params, global_updated = push_and_pull_gradients(grads, params, loss_fn_builder, accumulation_steps=100, lie_key=lie_subkey)
            if global_updated:
                print(f"[Step {step}] Update. Final Scan Loss: {float(loss):.5f}")
                step += 1
        except Exception as e: 
            print(e)
            time.sleep(1)
