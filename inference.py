import os, sys, glob, pickle, jax, json, random, fcntl
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wav

def load_checkpoint(path):
    with open(path, "rb") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        params = pickle.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return params

def get_seed(metadata, sr=44100, seq_len=20):
    while True:
        entry = random.choice(metadata)
        path = os.path.join("data", entry["shard"])
        if not os.path.exists(path): continue
        mmap = np.memmap(path, dtype=np.float32, mode='r').reshape(-1, 4)
        if (entry["num_samples"] / entry["sample_rate"]) <= seq_len: continue
        
        start = int(random.uniform(0, (entry["num_samples"] / entry["sample_rate"]) - seq_len) * entry["sample_rate"])
        off = entry["offset_bytes"] // 16
        return [mmap[off + start + (i*sr) : off + start + (i+1)*sr].flatten() / 32768.0 for i in range(seq_len)]

def run():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    steps, sr = int(np.ceil(dur)), 44100
    
    params = load_checkpoint(sorted(glob.glob("checkpoints/checkpoint_run.pickle"), key=os.path.getmtime)[-1])
    with open("checkpoints/model_jaxpr.pickle", "rb") as f: jaxpr = pickle.load(f)
    with open("data/audio_vault.meta.jsonl", "r") as f: meta = [json.loads(l) for l in f if l.strip()]

    ctx = jnp.expand_dims(jnp.array(get_seed(meta)), axis=0)
    gen = []
    
    for i in range(steps):
        nxt = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params, ctx)[0][:, -1, :]
        gen.append(np.array(nxt[0]))
        ctx = jnp.concatenate([ctx[:, 1:, :], jnp.expand_dims(nxt, axis=1)], axis=1)
        print(f"Step {i+1}/{steps}")

    wav_data = (np.concatenate(gen)[:int(dur * sr)] * 32768.0).clip(-32768, 32767).astype(np.int16)
    wav.write(f"vocals_{int(dur)}s.wav", sr, wav_data[:, :2])
    wav.write(f"inst_{int(dur)}s.wav", sr, wav_data[:, 2:])

if __name__ == "__main__": run()
