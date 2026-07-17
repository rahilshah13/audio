import os, glob, pickle, jax, optax, time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from jax.flatten_util import ravel_pytree

class SpectralPreconditionerMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.Dense(512)(x))
        x = nn.gelu(nn.Dense(512)(x))
        return jax.nn.sigmoid(nn.Dense(x.shape[-1])(x)) * 2.0 

class MetaDashboard:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.losses = []
    
    def update(self, loss):
        self.losses.append(loss)
        self.ax.clear()
        self.ax.plot(self.losses, color='#8b5cf6', label='Meta-Loss (Preconditioner MSE)')
        self.ax.set_title("Spectral Preconditioner Fidelity")
        plt.draw(); plt.pause(0.01)

def get_meta_preconditioner(grads):
    meta_ckpt = "checkpoints/meta_preconditioner.pickle"
    if not os.path.exists(meta_ckpt): return None
    
    ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
    if not ntk_files: return None
    
    ntk_data = jnp.array(jnp.load(ntk_files[-1]).flatten()[:1024])
    with open(meta_ckpt, "rb") as f: meta_params = pickle.load(f)
    
    model = SpectralPreconditionerMLP()
    scales = model.apply(meta_params, ntk_data)
    
    flat_grads, treedef = ravel_pytree(grads)
    scaled_flat = flat_grads * jax.image.resize(scales, (flat_grads.shape[0],), 'linear')
    return treedef(scaled_flat)

@jax.jit
def train_step(state, ntk_input, target_grads_flat):
    def loss_fn(params):
        pred_scales = SpectralPreconditionerMLP().apply(params, ntk_input)
        return jnp.mean(jnp.square(pred_scales - jnp.ones_like(pred_scales)))
    
    loss, grads = jax.value_and_grad(loss_fn)(state['params'])
    updates, new_opt_state = state['tx'].update(grads, state['opt_state'])
    return loss, {'params': optax.apply_updates(state['params'], updates), 
                  'opt_state': new_opt_state, 'tx': state['tx']}

def run_meta_daemon():
    print("[META-DAEMON] Initializing Spectral Preconditioner System...")
    dashboard = MetaDashboard()
    state = None 
    
    while True:
        ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
        if len(ntk_files) > 0:
            if state is None:
                dummy_ntk = jnp.zeros(1024)
                model = SpectralPreconditionerMLP()
                meta_params = model.init(jax.random.PRNGKey(0), dummy_ntk)
                tx = optax.adam(1e-4)
                state = {'params': meta_params, 'opt_state': tx.init(meta_params), 'tx': tx}
            
            ntk_data = jnp.load(ntk_files[-1]).flatten()[:1024]
            loss, state = train_step(state, ntk_data, None)
            
            dashboard.update(float(loss))
            with open("checkpoints/meta_preconditioner.pickle", "wb") as f:
                pickle.dump(state['params'], f)
        time.sleep(5)

if __name__ == "__main__":
    run_meta_daemon()
