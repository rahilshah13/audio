import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

SCALE_MODES = {"Major": [0, 2, 4, 5, 7, 9, 11], "Minor": [0, 2, 3, 5, 7, 8, 10]}
PITCH_CLASSES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

def hz_to_note(hz):
    if hz < 16: return "Noise"
    A4 = 440.0
    h = round(12 * np.log2(hz / A4))
    return f"{PITCH_CLASSES[h % 12]}{int(4 + (h + 9) // 12)}"

def analyze_acoustic_tokens(batch_waveform, sr=44100):
    freqs, notes = [], []
    # Process using the vocal channels (0 and 1) for musical analysis
    for token_idx in range(batch_waveform.shape[0]):
        channel_0 = batch_waveform[token_idx, ::4] # Extracting V_L
        fft_data = np.abs(np.fft.rfft(channel_0))
        fft_freqs = np.fft.rfftfreq(len(channel_0), d=1.0/sr)
        peak_idx = np.argmax(fft_data[1:]) + 1
        dom_freq = fft_freqs[peak_idx]
        freqs.append(dom_freq)
        notes.append(hz_to_note(dom_freq))
    return freqs, notes

def detect_musical_scale(notes):
    cleaned_pitches = {n[:-1] for n in notes if n != "Noise" and n[-1].isdigit()}
    if not cleaned_pitches: return "Unknown"
    best_scale, max_matches = "Chromatic", -1
    for root_idx, root in enumerate(PITCH_CLASSES):
        for mode_name, intervals in SCALE_MODES.items():
            scale_pitches = {PITCH_CLASSES[(root_idx + i) % 12] for i in intervals}
            matches = len(cleaned_pitches.intersection(scale_pitches))
            if matches > max_matches:
                max_matches = matches
                best_scale = f"{root} {mode_name}"
    return best_scale

def estimate_bpm(waveform, sr=44100):
    # Use channels 0 and 2 (V_L + I_L) to estimate rhythmic envelope
    envelope = np.abs(waveform[::100, 0] + waveform[::100, 2])
    env_mean = np.mean(envelope)
    if env_mean < 1e-4: return 120.0
    
    centered = envelope - env_mean
    corr = np.correlate(centered, centered, mode='full')[len(centered)-1:]
    
    min_lag = int(sr * 60 / (100 * 180))
    max_lag = int(sr * 60 / (100 * 60))
    
    if min_lag >= len(corr) or max_lag >= len(corr) or min_lag == max_lag:
        return float(75 + int(np.var(waveform) * 1e5) % 95)
        
    peak_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    bpm = (sr / 100) * 60 / peak_lag
    return float(np.clip(bpm, 60.0, 180.0))

class TrainingDashboard:
    def __init__(self, total_steps, num_heads=16, seq_len=20):
        self.total_steps = total_steps
        self.current_active_head = 0
        self.seq_len = seq_len
        self.history_records = [["--", "--", "--"] for _ in range(8)]        
        self.ntk_steps = []
        self.ntk_history = []
        plt.ion()
        
        self.fig = plt.figure(figsize=(26, 7.5))
        gs = self.fig.add_gridspec(2, 7, width_ratios=[1.1, 1.1, 1.1, 1.3, 1.3, 0.05, 1.5], height_ratios=[1.0, 1.0])
        plt.subplots_adjust(bottom=0.18, top=0.86, wspace=0.52, hspace=0.45, left=0.03, right=0.97)
        self.stat_text_obj = self.fig.suptitle(
            "CALM Training Dashboard | Step: 0 | Abs Loss: -- | Noise Scale: --",
            fontsize=11, fontweight='bold', y=0.96
        )

        ax_attn = self.fig.add_subplot(gs[:, 0])
        ax_alignment = self.fig.add_subplot(gs[:, 1])
        ax_batch_variance = self.fig.add_subplot(gs[:, 2]) 
        ax_ntk = self.fig.add_subplot(gs[:, 3])
        ax_loss = self.fig.add_subplot(gs[:, 4])
        ax_split = self.fig.add_subplot(gs[:, 5])
        ax_spectral = self.fig.add_subplot(gs[0, 6])  
        ax_registry = self.fig.add_subplot(gs[1, 6])  
        self.axs = [ax_attn, ax_alignment, ax_batch_variance, ax_ntk, ax_loss, ax_split, ax_spectral, ax_registry]

        attn_placeholder = np.zeros((seq_len, seq_len))
        self.heatmap = self.axs[0].imshow(attn_placeholder, vmin=0, vmax=1, cmap="magma", origin='lower')
        self.axs[0].set_ylabel("Query Token Index", fontsize=8, labelpad=4)
        self.axs[0].set_xlabel("Key Token Index", fontsize=8, labelpad=4)
        self.axs[0].set_xticks(np.arange(0, seq_len, 4))
        self.axs[0].set_yticks(np.arange(0, seq_len, 4))
        self.fig.colorbar(self.heatmap, ax=self.axs[0], fraction=0.046, pad=0.04)
        self._update_head_title()
        
        alignment_placeholder = np.zeros((seq_len, seq_len))
        self.alignment_heatmap = self.axs[1].imshow(alignment_placeholder, vmin=-1, vmax=1, cmap="coolwarm", origin='lower')
        self.axs[1].set_title("Profile Feature Alignment", fontsize=9, pad=18, fontweight='bold')
        self.axs[1].text(0.5, 1.04, "Attention Cosine Similarity", transform=self.axs[1].transAxes, ha='center', fontsize=7, color='#475569')
        self.axs[1].set_xlabel("Query Token Index", fontsize=8)
        self.axs[1].set_ylabel("Query Token Index", fontsize=8)
        self.axs[1].set_xticks(np.arange(0, seq_len, 5))
        self.axs[1].set_yticks(np.arange(0, seq_len, 5))
        self.fig.colorbar(self.alignment_heatmap, ax=self.axs[1], fraction=0.046, pad=0.04)

        batch_placeholder = np.zeros((seq_len, seq_len))
        self.batch_heatmap = self.axs[2].imshow(batch_placeholder, vmin=0, vmax=0.25, cmap="viridis", origin='lower')
        self.axs[2].set_title("Cross-Batch Attention Variance", fontsize=9, pad=18, fontweight='bold')
        self.axs[2].text(0.5, 1.04, "Variance calculated across all batch items", transform=self.axs[2].transAxes, ha='center', fontsize=7, color='#475569')
        self.axs[2].set_xlabel("Key Token Index", fontsize=8)
        self.axs[2].set_ylabel("Query Token Index", fontsize=8)
        self.axs[2].set_xticks(np.arange(0, seq_len, 5))
        self.axs[2].set_yticks(np.arange(0, seq_len, 5))
        self.fig.colorbar(self.batch_heatmap, ax=self.axs[2], fraction=0.046, pad=0.04)

        self.ntk_line, = self.axs[3].plot([], [], color='#8b5cf6', linewidth=1.8, marker='o', markersize=2)
        self.axs[3].set_title("Parameter NTK Evolution", fontsize=9, pad=18, fontweight='bold')
        self.axs[3].text(0.5, 1.04, "Global Representation Change", transform=self.axs[3].transAxes, ha='center', fontsize=7, color='#64748b', style='italic')
        self.axs[3].set_xlabel("Training Step", fontsize=8)
        self.axs[3].set_ylabel(r"$\Delta\Theta$", fontsize=8)
        self.axs[3].grid(True, ls=":", alpha=0.4)

        self.loss_line, = self.axs[4].plot([], [], color='#ef4444', linewidth=1.5, label='Abs Loss')
        self.axs[4].set_title("Optimization Trajectory", fontsize=9, pad=18, fontweight='bold')
        self.axs[4].set_xlabel("Training Step", fontsize=8)
        self.axs[4].set_ylabel("Absolute Batch Loss", fontsize=8, color='#ef4444')
        self.axs[4].grid(True, ls=":", alpha=0.4)

        self.noise_ax = self.axs[4].twinx()
        self.noise_line, = self.noise_ax.plot([], [], color='#0ea5e9', linewidth=1.0, alpha=0.5, linestyle=':', label='Noise Scale')
        self.noise_ax.set_ylabel("Injected Noise Scale", fontsize=8, color='#0ea5e9')

        self.axs[5].axis('off')
        self.axs[5].axvline(x=0.5, color='#cbd5e1', linestyle='--', linewidth=1.2)

        bar_positions = np.arange(seq_len)
        self.freq_bars = self.axs[6].bar(bar_positions, np.ones(seq_len)*10, color='#2cb2cb', edgecolor='black', alpha=0.8)
        self.axs[6].set_yscale('log')
        self.axs[6].set_ylim(10, 22050)
        self.axs[6].set_xlim(-0.5, seq_len - 0.5)
        self.spectral_title_obj = self.axs[6].set_title("Acoustic Token Spectral Peaks", fontsize=9, pad=18, fontweight='bold')
        self.spectral_sub_obj = self.axs[6].text(0.5, 1.04, "", transform=self.axs[6].transAxes, ha='center', fontsize=7, color='#475569')
        self.axs[6].set_xlabel("Sequence Step (Vocal Tokens)", fontsize=8)
        self.axs[6].set_ylabel("Dominant Peak Frequency (Hz)", fontsize=8)
        self.axs[6].grid(True, which="both", ls=":", alpha=0.4)

        self.axs[7].axis('off')
        self.axs[7].set_title("Historical Stream Registry", fontsize=9, pad=6, fontweight='bold')
        headers = ["Source Track ID (Window)", "Scale Mode", "BPM"]
        self.ui_table = self.axs[7].table(cellText=self.history_records, colLabels=headers, loc='center', cellLoc='center')
        self.ui_table.auto_set_font_size(False)
        self.ui_table.set_fontsize(7)
        self.ui_table.scale(1.0, 1.1)

        for i in [0, 1, 2, 3, 4, 6]:
            self.axs[i].tick_params(axis='both', which='major', labelsize=7)
            if i not in [3, 4]: self._strip_spines(self.axs[i])
        self.noise_ax.tick_params(axis='y', which='major', labelsize=7)

        self.token_labels = [self.axs[6].text(i, 12, '', ha='center', va='bottom', fontsize=7, rotation=90) for i in range(seq_len)]
        self.buttons = []
        self._setup_buttons(num_heads)
        self.fig.canvas.draw_idle()

    def _strip_spines(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _update_head_title(self):
        self.axs[0].set_title(f"Attention Map (H{self.current_active_head})", fontsize=9, pad=18, fontweight='bold')

    def _setup_buttons(self, num_heads):
        def make_callback(h_idx):
            def change_head(event):
                self.current_active_head = h_idx
                self._update_head_title()
                self.fig.canvas.draw_idle()
            return change_head

        for idx in range(num_heads):
            # Dynamic layout for 16 heads
            x_pos = 0.03 + (idx % 8) * 0.04
            y_pos = 0.03 if idx < 8 else 0.07
            ax_btn = plt.axes([x_pos, y_pos, 0.035, 0.03])
            btn = Button(ax_btn, f"H{idx}", color='#f8fafc', hovercolor='#e2e8f0')
            btn.on_clicked(make_callback(idx))
            self.buttons.append(btn)

    def _update_embedded_table_data(self):
        cells = self.ui_table.get_celld()
        for row_idx, data_row in enumerate(self.history_records):
            for col_idx, text_val in enumerate(data_row):
                cells[(row_idx + 1, col_idx)].get_text().set_text(text_val)

    def update(self, step, loss_val, noise_scale, seen_count, weights_tensor, raw_visual_waveform, sample_title, window_start_sec, current_ntk=None, initial_ntk=None, loss_history=None, step_history=None, noise_history=None):
        weights_np = np.array(weights_tensor)
        active_weights = weights_np[0, self.current_active_head, :, :]
        self.heatmap.set_data(active_weights)
        
        centered_weights = active_weights - np.mean(active_weights, axis=-1, keepdims=True)
        norm_centered = centered_weights / (np.linalg.norm(centered_weights, axis=-1, keepdims=True) + 1e-8)
        self.alignment_heatmap.set_data(np.dot(norm_centered, norm_centered.T))
        
        all_batch_head_weights = weights_np[:, self.current_active_head, :, :]
        batch_variance = np.var(all_batch_head_weights, axis=0) 
        self.batch_heatmap.set_data(batch_variance)
        
        frequencies, musical_notes = analyze_acoustic_tokens(raw_visual_waveform)
        detected_scale = detect_musical_scale(musical_notes)
        estimated_bpm = estimate_bpm(raw_visual_waveform)
        
        clean_track_id = sample_title.split('=')[-1][:8] if '=' in sample_title else "Track"
        source_label = f"{clean_track_id} ({window_start_sec}s)"
        
        self.history_records.pop(0)
        self.history_records.append([source_label, detected_scale, f"{estimated_bpm:.0f}"])
        
        for bar, freq, note_str, txt_obj in zip(self.freq_bars, frequencies, musical_notes, self.token_labels):
            safe_freq = max(freq, 10)
            bar.set_height(safe_freq)
            txt_obj.set_text(note_str)
            txt_obj.set_y(safe_freq * 1.15 if safe_freq > 20 else 12)
        
        if current_ntk is not None and initial_ntk is not None:
            deviation = np.linalg.norm(current_ntk - initial_ntk) / (np.linalg.norm(initial_ntk) + 1e-8)
            self.ntk_steps.append(step)
            self.ntk_history.append(deviation)
            self.ntk_line.set_data(self.ntk_steps, self.ntk_history)
            self.axs[3].set_xlim(min(self.ntk_steps), max(self.ntk_steps) + 1)
            self.axs[3].set_ylim(-0.005, max(self.ntk_history) * 1.2 + 0.01)

        if loss_history is not None and step_history is not None:
            self.loss_line.set_data(step_history, loss_history)
            self.axs[4].set_xlim(min(step_history), max(step_history) + 1)
            self.axs[4].set_ylim(min(loss_history) * 0.9, max(loss_history) * 1.1 + 0.01)
            
            if noise_history is not None:
                self.noise_line.set_data(step_history, noise_history)
                self.noise_ax.set_ylim(0.0, max(noise_history) * 1.2 + 0.01)

        self.spectral_sub_obj.set_text(f"Global Scale: {detected_scale} | Boundary: [{window_start_sec}s]")
        self.stat_text_obj.set_text(f"CALM Dashboard | Step: {step}/{self.total_steps} | Loss: {loss_val:.4f} | Noise: {noise_scale:.3f} | Streams: {seen_count}")
        
        self._update_embedded_table_data()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
