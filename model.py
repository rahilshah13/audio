import os, json, pickle, jax, optax, random, time, numpy as np
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from dashboard import TrainingDashboard
from meta import get_calm_params_from_ntk_trajectory

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
        # Output 176400 (Dual-stem: 88200 Vocals + 88200 Instrumentals)
        out = nn.Dense(176400, name="up_proj_2")(nn.gelu(nn.Dense(8192, name="up_proj_1")(nn.LayerNorm(name="ln_2")(h + ff))))
        if return_attn: return out, attn_weights
        return out

def normalize_loss(loss, scale=0.5):
    return 1.0 - np.exp(-loss / scale)

def sharded_memmap_loader(batch_size, seq_len=10, samples_per_sec=44100):
    meta_path = "data/audio_vault.meta.jsonl"
    with open(meta_path, "r") as f: metadata = [json.loads(l) for l in f if l.strip()]
    mmap_pool = {}
    while True:
        batch, batch_urls, start_seconds = [], set(), []
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            if entry["shard"] not in mmap_pool:
                # Assuming 4-channel interleaved in storage: [V_L, V_R, I_L, I_R]
                mmap_pool[entry["shard"]] = np.memmap(os.path.join("data", entry["shard"]), dtype=np.float32, mode='r').reshape(-1, 4)
            
            chosen_start = random.uniform(0, (entry["num_samples"] / entry["sample_rate"]) - seq_len)
            latents = []
            for i in range(seq_len):
                s_idx = (entry["offset_bytes"] // 16) + int((chosen_start + i) * entry["sample_rate"])
                chunk = mmap_pool[entry["shard"]][s_idx : s_idx + samples_per_sec]
                latents.append(chunk.flatten())
            batch.append(jnp.stack(latents))
            batch_urls.add(entry["url"])
        yield jnp.stack(batch), batch_urls, int(chosen_start)

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
    
    params = get_calm_params_from_ntk_trajectory(multiplier=1.0) or (pickle.load(open(checkpoint_path, "rb")) if os.path.exists(checkpoint_path) else model.init(key, jnp.zeros((1, 10, 176400)))['params'])

    ntk_calculator = make_ntk_fn(model)
    current_true_ntk = np.array(ntk_calculator(params, jax.random.normal(key, (1, 1, 176400))))

    MICRO_BATCH_SIZE, ACCUMULATION_STEPS = 1, 1000
    tx = optax.adam(2e-4)
    opt_state = tx.init(params)
    loader = sharded_memmap_loader(batch_size=MICRO_BATCH_SIZE)

    @partial(jax.jit, static_argnames=['noise_scale'])
    def micro_step(params, batch, key, noise_scale):
        noised = batch + jax.random.normal(jax.random.split(key)[0], batch.shape) * noise_scale
        preds = model.apply({'params': params}, noised[:, :-1, :])
        loss = jnp.mean(jnp.square(preds - batch[:, 1:, :]))
        return loss, jax.grad(lambda p: jnp.mean(jnp.square(model.apply({'params': p}, noised[:, :-1, :]) - batch[:, 1:, :])))(params)

    board = TrainingDashboard(total_steps=100000)
    for step in range(1, 100001):
        accum_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        total_loss = 0.0
        for _ in range(ACCUMULATION_STEPS):
            b_data, b_urls, _ = next(loader)
            loss, grads = micro_step(params, b_data, key, 0.05)
            accum_grads = jax.tree_util.tree_map(lambda a, g: a + (g / ACCUMULATION_STEPS), accum_grads, grads)
            total_loss += (float(loss) / ACCUMULATION_STEPS)
        
        params = optax.apply_updates(params, tx.update(accum_grads, opt_state, params)[0])
        if step % 100 == 0:
            np.save(f"ntk_logs/ntk_step_{step}.npy", np.array(ntk_calculator(params, jax.random.normal(key, (1, 1, 176400)))))
            with open(checkpoint_path, "wb") as f: pickle.dump(params, f)
