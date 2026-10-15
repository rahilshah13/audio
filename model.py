import os, json, pickle, jax, optax, random, time, sys, numpy as np
import jax.numpy as jnp
from functools import partial
from meta import get_meta_preconditioner
import fcntl 

STATE_FILE = "data/global_state.json"
CURR_CKPT = "checkpoints/checkpoint_bundle.pickle"
PREV_CKPT = "checkpoints/checkpoint_bundle_prev.pickle"

def gpt_forward(params, x, scale, bpm, target_dim=88200, n_heads=16):
    x = jax.nn.gelu(x @ params['down_proj_1']) @ params['down_proj_2']
    B, T, C = x.shape
    
    s_emb = params['scale_emb'][scale]
    b_emb = bpm[:, None] @ params['bpm_proj']
    cond = jnp.expand_dims(s_emb + b_emb, 1)
    
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

def lie_group_perturbation(w, generator_perturbation, eps=1e-3):
    skew_w = 0.5 * (generator_perturbation - jnp.swapaxes(generator_perturbation, -1, -2))
    w_perturbed = w @ jax.scipy.linalg.exptm(eps * skew_w)
    return w_perturbed

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

def cauchy_binning_and_merge_deltas(base_params, adapted_params, gamma=0.1, num_bins=16):
    """
    Quantizes parameter deltas using non-uniform Cauchy-spaced bins to 
    gracefully preserve heavy-tailed outlier updates.
    """
    def q_merge(base_p, adapt_p):
        delta = adapt_p - base_p
        
        # 1. Map delta to probability space [0, 1] via Cauchy CDF
        cdf_vals = (1.0 / jnp.pi) * jnp.arctan(delta / gamma) + 0.5
        
        # 2. Quantize into discrete bins in probability space
        quantized_cdf = jnp.round(cdf_vals * num_bins) / num_bins
        quantized_cdf = jnp.clip(quantized_cdf, 1.0 / (2.0 * num_bins), 1.0 - (1.0 / (2.0 * num_bins)))
        
        # 3. Map back to parameter space using the Cauchy Inverse CDF
        quantized_delta = gamma * jnp.tan(jnp.pi * (quantized_cdf - 0.5))
        
        return base_p + quantized_delta

    return jax.tree_util.tree_map(q_merge, base_params, adapted_params)

def stream_jacobian_chunks(loss_fn, params, chunk_size=1024):
    dummy_out = loss_fn(params)
    flat_out, _ = jax.flatten_util.ravel_pytree(dummy_out)
    out_dim = flat_out.shape[0]
    
    for start_idx in range(0, out_dim, chunk_size):
        end_idx = min(start_idx + chunk_size, out_dim)
        
        def sub_loss_fn(p):
            res = loss_fn(p)
            f_res, _ = jax.flatten_util.ravel_pytree(res)
            return f_res[start_idx:end_idx]
            
        _, sub_vjp = jax.vjp(sub_loss_fn, params)
        chunk_len = end_idx - start_idx
        
        def scan_chunk_row(carry, i):
            unit_vector = jnp.zeros(chunk_len).at[i].set(1.0)
            grad_cotangent = sub_vjp(unit_vector)
            flat_g, _ = jax.flatten_util.ravel_pytree(grad_cotangent[0])
            return carry, flat_g

        _, chunk_jacobian_rows = jax.lax.scan(scan_chunk_row, None, jnp.arange(chunk_len))
        yield chunk_jacobian_rows
        del sub_vjp
        jax.clear_caches()

def push_and_pull_gradients(local_grads, loss_val, accumulation_steps=1000, ntk_fingerprint=None):
    grad_store_path = "data/shared_gradients.pickle"
    
    if ntk_fingerprint is not None:
        os.makedirs("ntk_logs", exist_ok=True)
        timestamp_id = int(time.time() * 1000)
        np.save(f"ntk_logs/ntk_step_{timestamp_id}.npy", np.array(ntk_fingerprint))

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
        with open(CURR_CKPT, "r+b") as pf:
            fcntl.flock(pf, fcntl.LOCK_EX)
            bundle = pickle.load(pf)
            params = bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle
            
            preconditioner = get_meta_preconditioner(shared_grads, loss=loss_val)
            if preconditioner:
                shared_grads = jax.tree_util.tree_map(lambda g, p: g * p, shared_grads, preconditioner)
            tx = optax.adam(2e-4)
            opt_state_path = "checkpoints/opt_state.pickle"
            opt_state = pickle.load(open(opt_state_path, "rb")) if os.path.exists(opt_state_path) else tx.init(params)
            updates, new_opt_state = tx.update(shared_grads, opt_state, params)
            
            new_params = optax.apply_updates(params, updates)
            # Replaced linear uniform quantization with Cauchy binning
            params = cauchy_binning_and_merge_deltas(params, new_params, gamma=0.1, num_bins=16)
            
            if os.path.exists(CURR_CKPT):
                if os.path.exists(PREV_CKPT): os.remove(PREV_CKPT)
                os.rename(CURR_CKPT, PREV_CKPT)
                
            new_bundle = {"params": params}
            pf.seek(0); pf.truncate(); pickle.dump(new_bundle, pf)
            pickle.dump(new_opt_state, open(opt_state_path, "wb"))
            fcntl.flock(pf, fcntl.LOCK_UN)
            return params, True
            
    with open(CURR_CKPT, "rb") as pf:
        fcntl.flock(pf, fcntl.LOCK_SH)
        bundle = pickle.load(pf)
        params = bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle
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

def run_cauchy_grid_search(params, b_data, b_scales, b_bpms, b_stems, key):
    gamma_candidates = [0.01, 0.1, 0.5, 1.0]
    bin_candidates = [8, 16, 32, 64]
    
    results = []
    print("\n--- Starting Cauchy Quantization Grid Search ---")
    
    for gamma in gamma_candidates:
        for num_bins in bin_candidates:
            start_time = time.time()
            
            def test_cauchy_merge(base_p, adapt_p):
                def q_merge(base_p, adapt_p):
                    delta = adapt_p - base_p
                    cdf_vals = (1.0 / jnp.pi) * jnp.arctan(delta / gamma) + 0.5
                    quantized_cdf = jnp.round(cdf_vals * num_bins) / num_bins
                    quantized_cdf = jnp.clip(quantized_cdf, 1.0 / (2.0 * num_bins), 1.0 - (1.0 / (2.0 * num_bins)))
                    quantized_delta = gamma * jnp.tan(jnp.pi * (quantized_cdf - 0.5))
                    return base_p + quantized_delta
                return jax.tree_util.tree_map(q_merge, base_p, adapt_p)

            def loss_fn(p):
                stem_cond = p['stem_emb'][b_stems]
                pred_target = gpt_forward(p, b_data[:, :-1, :], b_scales, b_bpms) + jnp.expand_dims(stem_cond, 1)
                return jnp.mean(jnp.square(pred_target - b_data[:, 1:, :]))

            _, grads = jax.value_and_grad(loss_fn)(params)
            tx = optax.adam(2e-4)
            opt_state = tx.init(params)
            updates, _ = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            quantized_params = test_cauchy_merge(params, new_params)
            final_loss = loss_fn(quantized_params)
            duration = time.time() - start_time
            
            results.append({"gamma": gamma, "bins": num_bins, "loss": float(final_loss)})
            print(f"Gamma: {gamma:<5} | Bins: {num_bins:<3} | Loss: {float(final_loss):.6f} | Time: {duration:.3f}s")
            
    best_config = min(results, key=lambda x: x["loss"])
    print(f"\nBest Configuration Found -> Gamma: {best_config['gamma']}, Bins: {best_config['bins']} (Loss: {best_config['loss']:.6f})\n")
    return best_config

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("ntk_logs", exist_ok=True)
    if not os.path.exists(CURR_CKPT):
        with open(CURR_CKPT, "wb") as f: pickle.dump({"params": init_params(key)}, f)
    with open(CURR_CKPT, "rb") as f: 
        bundle = pickle.load(f)
        params = bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle
    
    dummy_input_x = jnp.zeros((1, 10, 88200))
    dummy_scale = jnp.array([0], dtype=jnp.int32)
    dummy_bpm = jnp.array([120.0], dtype=jnp.float32)
    
    if "--jaxpr" in sys.argv:
        print(jax.make_jaxpr(lambda p, x, s, b: gpt_forward(p, x, s, b))(params, dummy_input_x, dummy_scale, dummy_bpm))
        sys.exit(0)

    loader = daemon_memmap_loader(batch_size=1)
    
    # Optional tuning run on startup if requested via flag
    if "--tune-quant" in sys.argv:
        sample_batch, sample_scales, sample_bpms, sample_stems = next(loader)
        run_cauchy_grid_search(params, sample_batch, sample_scales, sample_bpms, sample_stems, key)
    
    @partial(jax.jit, static_argnames=['noise_scale', 'max_inner_steps', 'tol', 'num_diffusion_steps'])
    def train_step_until_zero(params, batch, scales, bpms, stems, key, noise_scale, max_inner_steps=1000, tol=1e-7, num_diffusion_steps=10):
        k1, k2 = jax.random.split(key)
        step_indices = jax.random.randint(k1, shape=(batch.shape[0],), minval=0, maxval=num_diffusion_steps)
        
        t = (step_indices.astype(jnp.float32) + 1.0) / float(num_diffusion_steps)
        t = t[:, None, None]
        alpha_t = jnp.cos(t * jnp.pi / 2.0)
        sigma_t = jnp.sin(t * jnp.pi / 2.0)
        
        noise = jax.random.normal(k2, batch.shape)
        noised = alpha_t * batch + sigma_t * noise * noise_scale
        
        def loss_fn(p):
            stem_cond = p['stem_emb'][stems]
            pred_target = gpt_forward(p, noised[:, :-1, :], scales, bpms) + jnp.expand_dims(stem_cond, 1)
            return jnp.mean(jnp.square(pred_target - batch[:, 1:, :]))
            
        def condition_fun(state):
            p, step_count, loss = state
            return (step_count < max_inner_steps) & (loss > tol)

        def body_fun(state):
            p, step_count, loss = state
            current_loss, grads = jax.value_and_grad(loss_fn)(p)
            
            p_updated = {}
            for k_key, param_val in p.items():
                if param_val.ndim == 2 and param_val.shape[0] == param_val.shape[1]:
                    p_updated[k_key] = lie_group_perturbation(param_val, grads[k_key], eps=1e-4)
                else:
                    p_updated[k_key] = param_val - 1e-4 * grads[k_key]
            
            return p_updated, step_count + 1, current_loss

        initial_loss, _ = jax.value_and_grad(loss_fn)(params)
        final_params, final_steps, final_loss = jax.lax.while_loop(
            condition_fun, body_fun, (params, 0, initial_loss)
        )
        
        _, final_grads = jax.value_and_grad(loss_fn)(final_params)
        
        chunk_iterator = stream_jacobian_chunks(loss_fn, final_params, chunk_size=512)
        accumulated_jacobian_features = []
        for chunk in chunk_iterator:
            compressed_chunk = jnp.mean(chunk, axis=0)
            accumulated_jacobian_features.append(compressed_chunk)
            
        full_jacobian_vec = jnp.concatenate(accumulated_jacobian_features) if accumulated_jacobian_features else jnp.zeros(1024)
        jacobian_signature = jnp.pad(full_jacobian_vec, (0, max(0, 1024 - full_jacobian_vec.shape[0])))[:1024]
        
        return final_loss, final_grads, jacobian_signature

    step = 1
    while True:
        try:
            b_data, b_scales, b_bpms, b_stems = next(loader)
            loss, grads, jac_sig = train_step_until_zero(params, b_data, b_scales, b_bpms, b_stems, key, 0.05)
            params, global_updated = push_and_pull_gradients(grads, loss_val=float(loss), accumulation_steps=100, ntk_fingerprint=jac_sig)
            if global_updated:
                print(f"[Step {step}] Update. Converged Loss: {float(loss):.5f}")
                step += 1
        except Exception as e: 
            print(e)
            time.sleep(1)
