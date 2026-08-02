import os, json, pickle, sys, jax, optax, random, time, numpy as np, contextlib
import jax.numpy as jnp
from functools import partial
from meta import get_meta_preconditioner
import fcntl

STATE_FILE = "data/global_state.json"
MAX_PROCESSED_WINDOWS = 50000
WINDOW_LOCK_FILE = "data/global_state.lock"
UPDATE_LOCK_FILE = "data/update.lock"

def rsqrt(x):
    return 1.0 / jnp.sqrt(x)

@jax.jit
def gpt_forward(params, x, scale, bpm, stems, latent_dim=1024, n_heads=16):
    x = jax.nn.gelu(x @ params['down_proj_1']) @ params['down_proj_2']
    B, T, C = x.shape
    cond = jnp.expand_dims(params['scale_emb'][scale] + bpm[:, None] @ params['bpm_proj'] + params['stem_emb'][stems], 1)
    x = x + cond
    head_dim = C // n_heads
    q, k, v = [(x @ params[k_name]).reshape(B, T, n_heads, head_dim).swapaxes(1, 2) for k_name in ('query', 'key', 'value')]
    scores = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :], (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim), -1e9)
    h = (jax.nn.softmax(scores, axis=-1) @ v).swapaxes(1, 2).reshape(B, T, C) + x
    h = (h - h.mean(-1, keepdims=True)) * rsqrt(h.var(-1, keepdims=True) + 1e-5) * params['ln1_scale'] + params['ln1_bias']
    h_norm = (h + jax.nn.gelu(h @ params['ff_1']) @ params['ff_2'])
    h_norm = (h_norm - h_norm.mean(-1, keepdims=True)) * rsqrt(h_norm.var(-1, keepdims=True) + 1e-5) * params['ln2_scale'] + params['ln2_bias']
    return jax.nn.gelu(h_norm @ params['up_proj_1']) @ params['up_proj_2']

def init_params(key, dim=1024, latent_dim=1024):
    keys = jax.random.split(key, 16)
    specs = [('down_proj_1', (dim, 2048)), ('down_proj_2', (2048, dim)), ('query', (dim, dim)),
             ('key', (dim, dim)), ('value', (dim, dim)), ('ff_1', (dim, dim * 4)),
             ('ff_2', (dim * 4, dim)), ('up_proj_1', (dim, 2048)), ('up_proj_2', (2048, latent_dim)),
             ('scale_emb', (128, dim)), ('bpm_proj', (1, dim)), ('stem_emb', (2, dim))]
    p = {k: jax.random.normal(keys[i], shape) * 0.02 for i, (k, shape) in enumerate(specs)}
    p.update({'ln1_scale': jnp.ones((dim,)), 'ln1_bias': jnp.zeros((dim,)),
              'ln2_scale': jnp.ones((dim,)), 'ln2_bias': jnp.zeros((dim,))})
    return p

@contextlib.contextmanager
def exclusive_lock(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield f
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def atomic_json_dump(obj, path):
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)

def atomic_pickle_dump(obj, path):
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, path)

def register_global_window(window_str):
    os.makedirs("data", exist_ok=True)
    with exclusive_lock(WINDOW_LOCK_FILE):
        try:
            with open(STATE_FILE, "r") as f: data = json.load(f)
        except Exception: data = {"processed_windows": []}
        windows = data.get("processed_windows", [])
        if window_str in windows: return False
        windows.append(window_str)
        if len(windows) > MAX_PROCESSED_WINDOWS: windows = windows[-MAX_PROCESSED_WINDOWS:]
        data["processed_windows"] = windows
        atomic_json_dump(data, STATE_FILE)
        return True

@jax.jit
def quantize_and_merge_deltas(base_params, adapted_params, key, sparsity_threshold=0.01):
    def q_merge(base_p, adapt_p, subkey):
        delta = adapt_p - base_p
        mag = jnp.abs(delta)
        flat_mag = mag.reshape(-1)
        k = max(1, int(flat_mag.shape[0] * (1.0 - sparsity_threshold)))
        thresh = jax.lax.top_k(flat_mag, k)[0][-1]
        scaled = jnp.where(mag >= thresh, delta, 0.0) * 100.0
        floored = jnp.floor(scaled)
        stoch = floored + (jax.random.uniform(subkey, scaled.shape) < (scaled - floored)).astype(jnp.float32)
        return base_p + stoch / 100.0
    subkeys = jax.random.split(key, len(jax.tree_util.tree_leaves(base_params)))
    return jax.tree_util.tree_map(q_merge, base_params, adapted_params, jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(base_params), subkeys))

def lie_group_perturbation_matching_update(params, shared_grads, noised_input, target_output, scales, bpms, stems,
                                            perturbation_scale=1e-4, key=None, loss=None):
    key = key if key is not None else jax.random.PRNGKey(0)
    flat_grads, _ = jax.flatten_util.ravel_pytree(shared_grads)
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    dxs = jax.random.normal(jax.random.split(key, 5)[0], (4, flat_params.shape[0])) * perturbation_scale
    
    loss_fn = lambda p_tree: jnp.mean(jnp.square(gpt_forward(p_tree, noised_input, scales, bpms, stems) - target_output))
    
    @jax.jit
    def grad_mapping(p_flat, dx):
        _, g_tree = jax.value_and_grad(loss_fn)(unflatten_fn(p_flat + dx))
        return jax.flatten_util.ravel_pytree(g_tree)[0]

    curvatures = [jnp.abs(jnp.dot(dx, flat_grads) / (jnp.dot(dx, jax.jvp(lambda p: grad_mapping(p, dx), (flat_params,), (dx,))[1]) + 1e-8)) for dx in dxs]
    clamped = jnp.clip(jnp.mean(jnp.array(curvatures)), 1.0, 2.0)
    
    # loss is passed through so the meta module can use it as a supervised
    # target / outcome signal instead of only seeing NTK statistics + drift.
    meta_precond = get_meta_preconditioner(shared_grads, loss=loss)
    factors = jnp.clip(jax.flatten_util.ravel_pytree(meta_precond)[0] * clamped, 1.0, 2.0) if meta_precond is not None else jnp.full_like(flat_grads, clamped)
    return unflatten_fn(flat_grads * factors)

def push_and_pull_gradients(local_grads, current_params, noised_input, target_output, scales, bpms, stems, loss,
                             accumulation_steps=100, lie_key=None, quant_key=None):
    grad_store = "data/shared_gradients.pickle"
    ckpt_dir = "checkpoints"
    ckpt_path, prev_ckpt_path = os.path.join(ckpt_dir, "checkpoint_bundle.pickle"), os.path.join(ckpt_dir, "checkpoint_bundle_prev.pickle")
    os.makedirs("data", exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

    with exclusive_lock(UPDATE_LOCK_FILE):
        try:
            with open(grad_store, "rb") as f: shared_data = pickle.load(f)
        except Exception: shared_data = {"accumulated_grads": None, "accumulated_loss": None, "count": 0}

        shared_data["accumulated_grads"] = local_grads if shared_data["accumulated_grads"] is None else jax.tree_util.tree_map(lambda x, y: x + y, shared_data["accumulated_grads"], local_grads)
        shared_data["accumulated_loss"] = float(loss) if shared_data.get("accumulated_loss") is None else shared_data["accumulated_loss"] + float(loss)
        shared_data["count"] += 1
        apply_update = shared_data["count"] >= accumulation_steps

        if apply_update:
            shared_grads = jax.tree_util.tree_map(lambda x: x / shared_data["count"], shared_data["accumulated_grads"])
            avg_loss = shared_data["accumulated_loss"] / shared_data["count"]
            shared_data = {"accumulated_grads": None, "accumulated_loss": None, "count": 0}
        
        atomic_pickle_dump(shared_data, grad_store)

        if not apply_update:
            try:
                with open(ckpt_path, "rb") as f: return pickle.load(f)["params"], False
            except Exception: return current_params, False

        try:
            with open(ckpt_path, "rb") as f: ckpt = pickle.load(f)
            params, opt_state = ckpt["params"], ckpt["opt_state"]
        except Exception:
            params, tx = current_params, optax.adamw(2e-4)
            opt_state = tx.init(params)

        if os.path.exists(ckpt_path): atomic_pickle_dump({"params": params, "opt_state": opt_state}, prev_ckpt_path)

        shared_grads = lie_group_perturbation_matching_update(params, shared_grads, noised_input, target_output, scales, bpms, stems, key=lie_key, loss=avg_loss)
        tx = optax.adamw(2e-4)
        updates, new_opt_state = tx.update(shared_grads, opt_state, params)
        params = quantize_and_merge_deltas(params, optax.apply_updates(params, updates), quant_key, sparsity_threshold=0.01)
        atomic_pickle_dump({"params": params, "opt_state": new_opt_state, "timestamp": time.time()}, ckpt_path)
        return params, True

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
            if entry["shard"] not in mmap_pool:
                mmap_pool[entry["shard"]] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1, 2)
            
            offset_frames = entry.get("offset_bytes", 0) // 8
            avail = mmap_pool[entry["shard"]].shape[0] - offset_frames
            req = seq_len * entry["sample_rate"]
            if avail <= req: continue
            
            start_idx = int(random.uniform(0, avail - req))
            stem_type = int(entry.get("stem", 0))
            if not register_global_window(f"{entry['shard']}:{start_idx}:stem_{stem_type}"): continue

            latents = [mmap_pool[entry["shard"]][offset_frames + start_idx + i * samples_per_sec : offset_frames + start_idx + (i + 1) * samples_per_sec].flatten() for i in range(seq_len)]
            batch.append(jnp.stack(latents))
            batch_scales.append(int(entry.get("scale", 0)))
            batch_bpms.append(float(entry.get("bpm", 120.0)))
            batch_stems.append(stem_type)

        yield jnp.stack(batch), jnp.array(batch_scales, dtype=jnp.int32), jnp.array(batch_bpms, dtype=jnp.float32), jnp.array(batch_stems, dtype=jnp.int32)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("ntk_logs", exist_ok=True)
    ckpt_path = "checkpoints/checkpoint_bundle.pickle"
    
    if not os.path.exists(ckpt_path):
        initial_params = init_params(key)
        tx = optax.adamw(2e-4)
        with exclusive_lock(UPDATE_LOCK_FILE):
            if not os.path.exists(ckpt_path):
                atomic_pickle_dump({"params": initial_params, "opt_state": tx.init(initial_params), "timestamp": time.time()}, ckpt_path)
                
    with open(ckpt_path, "rb") as f: params = pickle.load(f)["params"]
    loader = daemon_memmap_loader(batch_size=1)

    @partial(jax.jit, static_argnames=['noise_scale', 'num_diffusion_steps'])
    def train_step_optimized(params, batch, scales, bpms, stems, key, noise_scale, num_diffusion_steps=10):
        k1, k2, k_loop = jax.random.split(key, 3)
        t = ((jax.random.randint(k1, shape=(batch.shape[0],), minval=0, maxval=num_diffusion_steps).astype(jnp.float32) + 1.0) / float(num_diffusion_steps))[:, None, None]
        alpha_t, sigma_t = jnp.cos(t * jnp.pi / 2.0), jnp.sin(t * jnp.pi / 2.0)
        noised = alpha_t * batch + sigma_t * jax.random.normal(k2, batch.shape) * noise_scale
        
        loss_fn = lambda p: jnp.mean(jnp.square(gpt_forward(p, noised[:, :-1, :], scales, bpms, stems) - batch[:, 1:, :]))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads, k_loop, noised[:, :-1, :], batch[:, 1:, :]

    sample_batch, sample_scales, sample_bpms, sample_stems = next(loader)
    jaxpr_repr = jax.make_jaxpr(train_step_optimized)(params, sample_batch, sample_scales, sample_bpms, sample_stems, jax.random.split(key)[1], 0.05)
    with open("checkpoints/train_step.jaxpr", "wb") as f: pickle.dump(jaxpr_repr, f)

    if "--export-jaxpr" in sys.argv:
        print("JAXPR exported to checkpoints/train_step.jaxpr. Exiting.")
        sys.exit(0)

    step = 1
    while True:
        try:
            b_data, b_scales, b_bpms, b_stems = next(loader)
            key, subkey = jax.random.split(key)
            loss, grads, key, noised_input, target_output = train_step_optimized(params, b_data, b_scales, b_bpms, b_stems, subkey, 0.05)
            
            key, lie_subkey, quant_subkey = jax.random.split(key, 3)
            params, global_updated = push_and_pull_gradients(grads, params, noised_input, target_output, b_scales, b_bpms, b_stems, loss,
                                                              accumulation_steps=100, lie_key=lie_subkey, quant_key=quant_subkey)
            if global_updated:
                print(f"[Step {step}] Global Update Applied. Loss: {float(loss):.5f}")
                step += 1
        except Exception as e:
            print(e)
            time.sleep(1)
