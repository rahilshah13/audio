import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SCALE_MODES = {"Major": [0, 2, 4, 5, 7, 9, 11], "Minor": [0, 2, 3, 5, 7, 8, 10]}
PITCH_CLASSES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

def hz_to_note(hz):
    if hz < 16: return "Noise"
    A4 = 440.0
    h = round(12 * np.log2(hz / A4))
    return f"{PITCH_CLASSES[h % 12]}{int(4 + (h + 9) // 12)}"

def analyze_acoustic_tokens(batch_waveform, sr=44100):
    freqs, notes = [], []
    for token_idx in range(batch_waveform.shape[0]):
        channel_0 = batch_waveform[token_idx, ::4]
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
    def __init__(self, total_steps, num_heads=16, seq_len=20, output_dir="/app/dashboard_static"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_steps = total_steps
        self.current_active_head = 0
        self.seq_len = seq_len
        self.history_records = [["--", "--", "--"] for _ in range(8)]        
        self.ntk_steps, self.ntk_history = [], []
        
        self.fig = plt.figure(figsize=(26, 7.5))
        gs = self.fig.add_gridspec(2, 7, width_ratios=[1.1, 1.1, 1.1, 1.3, 1.3, 0.05, 1.5], height_ratios=[1.0, 1.0])
        plt.subplots_adjust(bottom=0.18, top=0.86, wspace=0.52, hspace=0.45, left=0.03, right=0.97)
        self.stat_text_obj = self.fig.suptitle("CALM Training Dashboard", fontsize=11, fontweight='bold', y=0.96)
        self.axs = [self.fig.add_subplot(gs[:, i]) for i in range(5)]
        self.axs.append(self.fig.add_subplot(gs[:, 5]))
        self.axs.append(self.fig.add_subplot(gs[0, 6]))
        self.axs.append(self.fig.add_subplot(gs[1, 6]))

        # Initialize plots (omitted redundant setup code for brevity)
        self.heatmap = self.axs[0].imshow(np.zeros((seq_len, seq_len)), vmin=0, vmax=1, cmap="magma")
        self.ntk_line, = self.axs[3].plot([], [], color='#8b5cf6')
        self.loss_line, = self.axs[4].plot([], [], color='#ef4444')
        self.freq_bars = self.axs[6].bar(np.arange(seq_len), np.ones(seq_len)*10, color='#2cb2cb')
        self.ui_table = self.axs[7].table(cellText=self.history_records, colLabels=["Track", "Scale", "BPM"], loc='center')

    def update(self, step, loss_val, noise_scale, seen_count, weights_tensor, raw_visual_waveform, sample_title, window_start_sec, **kwargs):
        # Data Update Logic
        weights_np = np.array(weights_tensor)
        self.heatmap.set_data(weights_np[0, self.current_active_head, :, :])
        
        # Spectral/Music Analysis
        freqs, notes = analyze_acoustic_tokens(raw_visual_waveform)
        for bar, freq in zip(self.freq_bars, freqs): bar.set_height(max(freq, 10))
        
        # Save frame to disk
        self.stat_text_obj.set_text(f"Step: {step} | Loss: {loss_val:.4f}")
        self.fig.savefig(os.path.join(self.output_dir, "dashboard.png"), dpi=100)
