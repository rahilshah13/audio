import os, glob, pickle, jax, optax, time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from jax.flatten_util import ravel_pytree

CURR_CKPT = "checkpoints/checkpoint_bundle.pickle"
PREV_CKPT = "checkpoints/checkpoint_bundle_prev.pickle"

def align_drift(w_new, w_old):
    w1, _ = ravel_pytree(w_new)
    w2, _ = ravel_pytree(w_old)
    return jnp.linalg.norm(w1 - w2) / (jnp.linalg.norm(w1) + 1e-6)

class SpectralPreconditionerMLP(nn.Module):
    fixed_dim: int = 1025

    @nn.compact
    def __call__(self, x, target_dim=None):
        if x.shape[-1] != self.fixed_dim:
            x = jax.image.resize(x, (self.fixed_dim,), 'linear')

        x = nn.Dense(self.fixed_dim)(x)
        x = nn.gelu(nn.Dense(512)(x))
        x = nn.gelu(nn.Dense(512)(x))
        scales = jax.nn.sigmoid(nn.Dense(self.fixed_dim)(x)) * 2.0

        if target_dim is not None and target_dim != self.fixed_dim:
            scales = jax.image.resize(scales, (target_dim,), 'linear')
        return scales

class MetaDashboard:
    def __init__(self):
        self.enabled = True
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 4))
            self.losses = []
        except Exception:
            self.enabled = False

    def update(self, loss):
        if not self.enabled:
            return
        try:
            self.losses.append(loss)
            self.ax.clear()
            self.ax.plot(self.losses, color='#8b5cf6', label='Meta-Loss (Curvature Variance)')
            self.ax.set_title("Manifold-Aware Spectral Preconditioner")
            plt.draw(); plt.pause(0.01)
        except Exception:
            self.enabled = False

def _load_params_from_bundle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["params"] if isinstance(obj, dict) and "params" in obj else obj

def get_meta_preconditioner(grads, loss=None):
    meta_ckpt = "checkpoints/meta_preconditioner.pickle"
    if not os.path.exists(meta_ckpt):
        return None

    ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
    if not ntk_files:
        return None

    drift = 0.0
    if os.path.exists(CURR_CKPT) and os.path.exists(PREV_CKPT):
        try:
            w_new = _load_params_from_bundle(CURR_CKPT)
            w_old = _load_params_from_bundle(PREV_CKPT)
            drift = align_drift(w_new, w_old)
        except Exception:
            pass

    loss_val = 0.0 if loss is None else float(loss)

    raw_jac = jnp.array(np.load(ntk_files[-1])).flatten()
    if raw_jac.shape[0] < 1024:
        raw_jac = jnp.pad(raw_jac, (0, 1024 - raw_jac.shape[0]))
    else:
        raw_jac = raw_jac[:1024]

    ntk_data = jnp.concatenate([raw_jac, jnp.array([drift, loss_val])])

    with open(meta_ckpt, "rb") as f:
        meta_params = pickle.load(f)

    flat_grads, treedef = ravel_pytree(grads)
    target_dim = flat_grads.shape[0]

    model = SpectralPreconditionerMLP()
    scales = model.apply(meta_params, ntk_data, target_dim=target_dim)

    scaled_flat = flat_grads * scales
    return treedef(scaled_flat)

@jax.jit
def train_step(params, opt_state, tx, inputs):
    def loss_fn(p):
        pred_scales = SpectralPreconditionerMLP().apply(p, inputs)
        jac_diag = inputs[:1024]
        effective_curvature = pred_scales[:jac_diag.shape[0]] * jac_diag
        mean_curv = jnp.mean(effective_curvature)
        return jnp.var(effective_curvature) + 0.1 * jnp.square(mean_curv - 1.0)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

def run_meta_daemon():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("ntk_logs", exist_ok=True)
    dashboard = MetaDashboard()
    params, opt_state, tx = None, None, optax.adam(1e-4)

    while True:
        ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
        if len(ntk_files) > 0:
            try:
                raw_jac = jnp.array(np.load(ntk_files[-1])).flatten()
                if raw_jac.shape[0] < 1024:
                    raw_jac = jnp.pad(raw_jac, (0, 1024 - raw_jac.shape[0]))
                else:
                    raw_jac = raw_jac[:1024]

                ntk_data = jnp.concatenate([raw_jac, jnp.array([0.0, 0.0])])

                if params is None:
                    dummy_input = jnp.zeros(1025)
                    params = SpectralPreconditionerMLP().init(jax.random.PRNGKey(0), dummy_input)
                    opt_state = tx.init(params)

                loss, params, opt_state = train_step(params, opt_state, tx, ntk_data)

                dashboard.update(float(loss))
                with open("checkpoints/meta_preconditioner.pickle", "wb") as f:
                    pickle.dump(params, f)
            except Exception:
                pass
        time.sleep(5)

if __name__ == "__main__":
    run_meta_daemon()
