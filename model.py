"""
SUMMARY:
Unified non-blocking CALM architecture featuring an optimized, fast-refreshing
Matplotlib visualization dashboard showing a true zero-centered empirical activation 
alignment profile per head (allowing blue tones) and second chunk window bounds in the title.
"""
import os, json, pickle, jax, optax, random, time, glob, re, tempfile
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Ensure an interactive, multi-threaded GUI backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from flax import linen as nn

# --- REFACTORED CORE MODEL SPECIFICATION ---
class CALM(nn.Module):
    dim: int = 1024
    
    @nn.compact
    def __call__(self, x, return_attn: bool = False):
        x = nn.Dense(self.dim, name="down_proj_2")(nn.gelu(nn.Dense(2048, name="down_proj_1")(x)))        
        B, T, C = x.shape
        num_heads = 8
        head_dim = self.dim // num_heads
        q = nn.Dense(self.dim, name="query")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        k = nn.Dense(self.dim, name="key")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
        v = nn.Dense(self.dim, name="value")(x).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)        
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
        tril = jnp.tril(jnp.ones((T, T), dtype=bool))
        mask = tril[None, None, :, :]  # Shape: (1, 1, T, T) to cleanly broadcast
        scores = jnp.where(mask, scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        h = jnp.matmul(attn_weights, v).swapaxes(1, 2).reshape(B, T, C)
        h = nn.Dense(self.dim, name="attn_out")(h)
        h = nn.LayerNorm(name="ln_1")(h + x)
        ff = nn.Dense(self.dim, name="ff_2")(nn.gelu(nn.Dense(self.dim * 2, name="ff_1")(h)))
        out = nn.Dense(88200, name="up_proj_2")(nn.gelu(nn.Dense(2048, name="up_proj_1")(nn.LayerNorm(name="ln_2")(h + ff))))
        
        if return_attn:
            return out, attn_weights
        return out

def hz_to_note(hz):
    if hz < 16: return "Noise"
    A4 = 440.0
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    h = round(12 * np.log2(hz / A4))
    return f"{notes[h % 12]}{int(4 + (h + 9) // 12)}"

def analyze_acoustic_tokens(batch_waveform, sr=44100):
    freqs, notes = [], []
    for token_idx in range(batch_waveform.shape[0]):
        channel_0 = batch_waveform[token_idx, ::2]  # Extract left audio channel
        fft_data = np.abs(np.fft.rfft(channel_0))
        fft_freqs = np.fft.rfftfreq(len(channel_0), d=1.0/sr)
        peak_idx = np.argmax(fft_data[1:]) + 1  # Ignore DC offset
        dom_freq = fft_freqs[peak_idx]
        freqs.append(dom_freq)
        notes.append(hz_to_note(dom_freq))
    return freqs, notes

# --- METADATA STREAMING ENGINE ---
def sharded_memmap_loader(batch_size=4, seq_len=20, samples_per_sec=44100):
    meta_path = "data/audio_vault.meta.jsonl"
    if not os.path.exists(meta_path):
        os.makedirs("data", exist_ok=True)
        with open(meta_path, "w") as f:
            f.write(json.dumps({"shard": "mock.bin", "num_samples": 8820000, "sample_rate": 44100, "offset_bytes": 0, "url": "mock_source"}) + "\n")
        with open("data/mock.bin", "wb") as f:
            f.write(np.random.randn(8820000, 2).astype(np.float32).tobytes())

    with open(meta_path, "r") as f: metadata = [json.loads(l) for l in f if l.strip()]
    mmap_pool = {}
    while True:
        batch, batch_urls, start_seconds = [], set(), []
        while len(batch) < batch_size:
            entry = random.choice(metadata)
            bin_path = os.path.join("data", entry["shard"])
            if entry["shard"] not in mmap_pool:
                mmap_pool[entry["shard"]] = np.memmap(bin_path, dtype=np.float32, mode='r').reshape(-1, 2)
            
            total_sec = entry["num_samples"] / entry["sample_rate"]
            latents = []
            chosen_start_offset = random.uniform(0, total_sec - seq_len)
            if len(start_seconds) == 0:  # Track index 0 for visualization
                start_seconds.append(int(chosen_start_offset))
                
            for i in range(seq_len):
                s_idx = (entry["offset_bytes"] // 8) + int((chosen_start_offset + i) * entry["sample_rate"])
                chunk = mmap_pool[entry["shard"]][s_idx : s_idx + samples_per_sec]
                if len(chunk) < samples_per_sec:
                    padded = np.zeros((samples_per_sec, 2), dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                latents.append(chunk.flatten())
            batch.append(jnp.stack(latents))
            batch_urls.add(entry["url"])
        yield jnp.stack(batch), batch_urls, start_seconds[0]

def loss_fn(params, x, key):
    noised = x + jax.random.normal(jax.random.split(key)[0], x.shape) * 0.02
    return jnp.mean(jnp.square(model.apply({'params': params}, noised[:, :-1, :]) - x[:, 1:, :]))

# --- COMPILING PIPELINES ---
model, key = CALM(), jax.random.PRNGKey(42)
os.makedirs("checkpoints", exist_ok=True)
params = model.init(key, jnp.zeros((1, 20, 88200)))['params']

tx = optax.adam(2e-4)
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# --- INITIALIZE MATPLOTLIB INTERACTIVE CONTROL DASHBOARD ---
plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(18, 6.0))
plt.subplots_adjust(bottom=0.22, top=0.88, wspace=0.32)
current_active_head = 0
global_seen_urls = set()
TOTAL_STEPS = 50000

stat_text_obj = fig.suptitle(
    "Initialization Mode | Step: 0 | Scaled Loss: -- | Unique Shards Covered: 0",
    fontsize=12, fontweight='bold', y=0.96
)

# Column 1: Clean Raw Attention Matrix
attn_matrix_placeholder = np.zeros((20, 20))
heatmap = axs[0].imshow(attn_matrix_placeholder, vmin=0, vmax=1, cmap="magma", origin='lower')
axs[0].set_title(f"Attention Matrix Profile (Head {current_active_head})", fontsize=10, pad=12, fontweight='bold')
axs[0].set_ylabel("Query Token Index (Target Context/Row Focus)", fontsize=9, labelpad=8)
axs[0].set_xlabel("Key Token Index (Source Context)", fontsize=9, labelpad=8)
axs[0].set_xticks(np.arange(20))
axs[0].set_yticks(np.arange(20))
fig.colorbar(heatmap, ax=axs[0], fraction=0.046, pad=0.04)

# Column 2: Dominant Spectral Energy Distribution
bar_positions = np.arange(20)
freq_bars = axs[1].bar(bar_positions, np.ones(20)*10, color='#2cb2cb', edgecolor='black', alpha=0.85)
axs[1].set_yscale('log')
axs[1].set_ylim(10, 22050)
axs[1].set_xlim(-0.5, 19.5)
spectral_title_obj = axs[1].set_title("Dominant Spectral Energy Distribution\n[Source: Initializing...]", fontsize=10, pad=12, fontweight='bold')

# Simple clean token frame index look
axs[1].set_xlabel("Token Frame Window Index", fontsize=9, labelpad=8)
axs[1].set_ylabel("Log Scale Frequency (Hz)", fontsize=9)
axs[1].grid(True, which="both", ls="--", alpha=0.3)
axs[1].set_xticks(np.arange(0, 21, 2))
axs[1].set_xticklabels([str(i) for i in range(0, 21, 2)])

# Column 3: Zero-Centered Feature Activation Alignment Map
alignment_matrix_placeholder = np.zeros((20, 20))
# CHANGED: Map dynamically scales from -1 to 1 to accommodate blue negative correlation structures
alignment_heatmap = axs[2].imshow(alignment_matrix_placeholder, vmin=-1, vmax=1, cmap="coolwarm", origin='lower')
axs[2].set_title("Empirical Feature Activation Alignment Profile", fontsize=10, pad=12, fontweight='bold')
axs[2].set_xlabel("Token Step Contrast Space", fontsize=9)
axs[2].set_xticks(np.arange(0, 20, 5))
axs[2].set_yticks(np.arange(0, 20, 5))
fig.colorbar(alignment_heatmap, ax=axs[2], fraction=0.046, pad=0.04)

for ax in [axs[0], axs[2]]:
    ax.tick_params(axis='both', which='major', labelsize=8)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

token_labels = [axs[1].text(i, 12, '', ha='center', va='bottom', fontsize=8, rotation=90, fontweight='medium') for i in range(20)]

buttons = []
def make_tab_callback(head_idx):
    def change_active_head(event):
        global current_active_head
        current_active_head = head_idx
        axs[0].set_title(f"Attention Matrix Profile (Head {current_active_head})", fontsize=10, pad=12, fontweight='bold')
    return change_active_head

for idx in range(8):
    ax_btn = plt.axes([0.38 + idx * 0.035, 0.04, 0.03, 0.035])
    btn = Button(ax_btn, f"H{idx}", color='#f0f0f0', hovercolor='#d0d0d0')
    btn.on_clicked(make_tab_callback(idx))
    buttons.append(btn)

plt.show(block=False)

# --- TRACKING VALS FOR DYNAMIC LOSS SCALING ---
max_observed_loss = -1e9
min_observed_loss = 1e9

# --- EXECUTION TRAJECTORY ---
loader = sharded_memmap_loader(batch_size=4, seq_len=20)
checkpoint_path = "checkpoints/checkpoint_run.pickle"

print("\nExecuting Training Iterations. Dashboard running on optimized UI pass loops...\n")

try:
    for step in range(1, TOTAL_STEPS + 1):
        current_global_step = step
        batch_data, step_urls, window_start_sec = next(loader)
        global_seen_urls.update(step_urls)
        key, step_key = jax.random.split(key)        
        params, opt_state, loss = train_step(params, opt_state, batch_data, step_key)        
        loss_val = float(loss)
        max_observed_loss = max(max_observed_loss, loss_val)
        min_observed_loss = min(min_observed_loss, loss_val)
        denom = (max_observed_loss - min_observed_loss)
        scaled_loss = (loss_val - min_observed_loss) / denom if denom > 1e-8 else 0.5
        
        if step % 10 == 0 or step == 1:
            print(f"\rProgress: {(step / TOTAL_STEPS) * 100:6.2f}% | Step {step}/{TOTAL_STEPS} | Norm Loss: {scaled_loss:.4f}", end="", flush=True)            
            
            _, weights_tensor = model.apply({'params': params}, batch_data, return_attn=True)            
            
            # Left Plot: Clean raw active attention weights
            active_weights = np.array(weights_tensor[0, current_active_head, :, :])
            heatmap.set_data(active_weights)
            
            # CHANGED: Center the weights around zero to capture inverse/negative relationships (introducing BLUE elements)
            centered_weights = active_weights - np.mean(active_weights, axis=-1, keepdims=True)
            norm_centered = centered_weights / (np.linalg.norm(centered_weights, axis=-1, keepdims=True) + 1e-8)
            visualized_alignment = np.dot(norm_centered, norm_centered.T)
            alignment_heatmap.set_data(visualized_alignment)
            
            # Middle Plot: Frequency feature structures
            raw_wave_sequence = np.array(batch_data[0])
            frequencies, musical_notes = analyze_acoustic_tokens(raw_wave_sequence)            
            
            for bar, freq, note_str, txt_obj in zip(freq_bars, frequencies, musical_notes, token_labels):
                safe_freq = max(freq, 10)
                bar.set_height(safe_freq)
                txt_obj.set_text(note_str)
                txt_obj.set_y(safe_freq * 1.1 if safe_freq > 20 else 12)
            
            # Embedded the exact second chunks into the title text string header directly
            current_sample_title = list(step_urls)[0] if step_urls else "Unknown Source"
            window_end_sec = window_start_sec + 20
            spectral_title_obj.set_text(
                f"Dominant Spectral Energy [{window_start_sec}s - {window_end_sec}s]\nSource: {current_sample_title}"
            )
            
            stat_text_obj.set_text(
                f"CALM Training Dashboard | Step: {step}/{TOTAL_STEPS} | Scaled Loss [0-1]: {scaled_loss:.4f} | Seen Sources: {len(global_seen_urls)}"
            )
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
        if step % 1000 == 0:
            with open(checkpoint_path, "wb") as f: 
                pickle.dump(params, f)
                
except KeyboardInterrupt:
    print("\nExecution safely interrupted by user request.")
finally:
    with open(checkpoint_path, "wb") as f: 
        pickle.dump(params, f)
    print("\nCheckpoints successfully synchronized with persistent volume structures.")