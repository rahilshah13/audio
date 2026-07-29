import os, sys, glob, pickle, json, random, fcntl
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav

def load_checkpoint(path):
    with open(path, "rb") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        params = pickle.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return params

def get_seed(metadata, sr=44100, seq_len=10):
    while True:
        entry = random.choice(metadata)
        path = os.path.join("data", entry["shard"])
        if not os.path.exists(path): continue
        # 2-channel audio layout (stride of 2 channels instead of 4)
        bytes_per_frame = 8  # 2 channels * 4 bytes per float32
        mmap = np.memmap(path, dtype=np.float32, mode='r').reshape(-1, 2)
        if (entry["num_samples"] / entry["sample_rate"]) <= seq_len: continue
        
        start = int(random.uniform(0, (entry["num_samples"] / entry["sample_rate"]) - seq_len) * entry["sample_rate"])
        off = entry["offset_bytes"] // bytes_per_frame
        latents = [
            mmap[off + start + (i * sr) : off + start + ((i + 1) * sr)].flatten()
            for i in range(seq_len)
        ]
        return latents

def run():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    steps, sr = int(np.ceil(dur)), 44100
    
    ckpt_files = glob.glob("checkpoints/checkpoint_run.pickle")
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint found at checkpoints/checkpoint_run.pickle")
    params = load_checkpoint(sorted(ckpt_files, key=os.path.getmtime)[-1])
    
    meta_path = "data/audio_vault.meta.jsonl"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    with open(meta_path, "r") as f: 
        meta = [json.loads(l) for l in f if l.strip()]

    # Import gpt_forward directly from model.py to ensure identical architecture and conditioning pathways
    from model import gpt_forward
    
    seed_latents = get_seed(meta)
    ctx = jnp.expand_dims(jnp.stack(seed_latents), axis=0)
    
    # Default conditioning parameters for inference execution
    scale = jnp.array([0], dtype=jnp.int32)
    bpm = jnp.array([120.0], dtype=jnp.float32)
    stems = jnp.array([0], dtype=jnp.int32)
    
    gen = []
    for i in range(steps):
        nxt = gpt_forward(params, ctx, scale, bpm, stems)
        nxt_slice = nxt[:, -1, :]
        gen.append(np.array(nxt_slice[0]))
        ctx = jnp.concatenate([ctx[:, 1:, :], jnp.expand_dims(nxt_slice, axis=1)], axis=1)
        print(f"Step {i+1}/{steps}")

    wav_data = (np.concatenate(gen)[:int(dur * sr)] * 32768.0).clip(-32768, 32767).astype(np.int16)
    
    # Reshape back to 2-channel stereo format matching target_dim layout (88200 values -> 44100 samples x 2 channels)
    stereo_output = wav_data.reshape(-1, 2)
    wav.write(f"output_stereo_{int(dur)}s.wav", sr, stereo_output)
    print(f"Successfully generated output_stereo_{int(dur)}s.wav")

if __name__ == "__main__": 
    run()
