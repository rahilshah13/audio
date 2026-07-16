import os, glob, pickle, jax, optax, time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from jax.flatten_util import ravel_pytree

# =================================================================
# 1. DECODER & DASHBOARD
# =================================================================
class NTKDecoderMLP(nn.Module):
    target_param_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.Dense(1024, name="h1")(x))
        x = nn.gelu(nn.Dense(4096, name="h2")(x))
        x = nn.gelu(nn.Dense(8192, name="h3")(x))
        return nn.Dense(self.target_param_dim, name="out")(x)

class MetaDashboard:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.losses = []
    
    def update(self, loss):
        self.losses.append(loss)
        self.ax.clear()
        self.ax.plot(self.losses, color='#8b5cf6', label='Meta-Loss (MSE)')
        self.ax.set_title("Meta-MLP Parameter Mapping Fidelity")
        self.ax.set_xlabel("Meta-Training Step")
        self.ax.set_ylabel("MSE (Predicted vs. CALM Weights)")
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

# =================================================================
# 2. META-LEARNING API (For model.py injection)
# =================================================================
def get_calm_params_from_ntk_trajectory(multiplier=1.0):
    meta_ckpt = "checkpoints/meta_mlp.pickle"
    calm_ckpt = "checkpoints/checkpoint_run.pickle"
    
    if not os.path.exists(meta_ckpt) or not os.path.exists(calm_ckpt):
        return None

    with open(calm_ckpt, "rb") as f:
        reference_params = pickle.load(f)
    _, treedef = ravel_pytree(reference_params)
    target_dim = len(ravel_pytree(reference_params)[0])

    with open(meta_ckpt, "rb") as f:
        meta_params = pickle.load(f)

    ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
    if len(ntk_files) < 2: return None
        
    ntk_t_minus_1 = np.load(ntk_files[-2]).flatten()
    ntk_t_final = np.load(ntk_files[-1]).flatten()
    
    # Project forward along the learned manifold
    projected_ntk = jnp.array(ntk_t_final + ((ntk_t_final - ntk_t_minus_1) * (multiplier - 1.0)))
    
    model = NTKDecoderMLP(target_param_dim=target_dim)
    predicted_flat = model.apply(meta_params, projected_ntk)
    
    return treedef(predicted_flat)

# =================================================================
# 3. TRAINING LOGIC (Daemon)
# =================================================================
@jax.jit
def train_step(state, ntk_input, target_flat):
    def loss_fn(params):
        pred = NTKDecoderMLP(target_param_dim=target_flat.shape[0]).apply(params, ntk_input)
        return jnp.mean(jnp.square(pred - target_flat))
    
    loss, grads = jax.value_and_grad(loss_fn)(state['params'])
    updates, new_opt_state = state['tx'].update(grads, state['opt_state'])
    new_params = optax.apply_updates(state['params'], updates)
    return loss, {'params': new_params, 'opt_state': new_opt_state, 'tx': state['tx']}

def run_meta_daemon():
    print("[META-DAEMON] Initializing Unified Meta-Training System...")
    dashboard = MetaDashboard()
    state = None 
    
    while True:
        ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
        ckpt_path = "checkpoints/checkpoint_run.pickle"
        
        if len(ntk_files) > 0 and os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                target_params = pickle.load(f)
            flat_target, _ = ravel_pytree(target_params)
            
            if state is None:
                print("[META-DAEMON] Initializing Meta-Architecture...")
                dummy_ntk = jnp.zeros(np.load(ntk_files[0]).flatten().shape)
                model = NTKDecoderMLP(target_param_dim=flat_target.shape[0])
                meta_params = model.init(jax.random.PRNGKey(0), dummy_ntk)
                tx = optax.adam(1e-4)
                state = {'params': meta_params, 'opt_state': tx.init(meta_params), 'tx': tx}
            
            ntk_data = np.load(ntk_files[-1]).flatten()
            loss, state = train_step(state, jnp.array(ntk_data), flat_target)
            
            step_name = ntk_files[-1].split('_')[-1].replace('.npy', '')
            print(f"[META-PROGRESS] NTK Step: {step_name} | Meta-MSE: {loss:.8f}")
            dashboard.update(float(loss))
            
            with open("checkpoints/meta_mlp.pickle", "wb") as f:
                pickle.dump(state['params'], f)
        
        time.sleep(5)

if __name__ == "__main__":
    run_meta_daemon()
