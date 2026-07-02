import os, json
import numpy as np
from scipy.io import wavfile
from yt_dlp import YoutubeDL

MAX_SHARD_BYTES = 500 * 1024 * 1024
DATA_DIR = "data"
URL_FILE = "data/urls.txt"
META_PATH = "data/audio_vault.meta.jsonl"

os.makedirs(DATA_DIR, exist_ok=True)

def get_current_shard_info():
    shard_idx = 0
    while True:
        bin_path = os.path.join(DATA_DIR, f"shard_{shard_idx}.bin")
        if not os.path.exists(bin_path):
            return shard_idx, bin_path, 0
        size = os.path.getsize(bin_path)
        if size < MAX_SHARD_BYTES:
            return shard_idx, bin_path, size
        shard_idx += 1

if os.path.exists(URL_FILE):
    with open(URL_FILE, "r+") as f:
        urls = f.read().splitlines()
        f.seek(0)
        
        for url in urls:
            if url.startswith("DONE: "):
                f.write(f"{url}\n")
                continue
                
            try:
                ydl_opts = {'format': 'bestaudio', 'outtmpl': 'data/%(id)s.%(ext)s', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}], 'quiet': True}
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    wav_path = os.path.join(DATA_DIR, f"{info['id']}.wav")
                
                sr, data = wavfile.read(wav_path)
                if data.ndim == 1: 
                    data = data[:, None].repeat(2, axis=1)
                
                data_clean = data.astype(np.float32)
                
                shard_idx, bin_path, current_bytes = get_current_shard_info()
                
                with open(bin_path, "ab") as bf:
                    bf.write(data_clean.tobytes())
                
                meta_entry = {
                    "shard": f"shard_{shard_idx}.bin",
                    "offset_bytes": current_bytes,
                    "num_samples": len(data_clean),
                    "sample_rate": sr,
                    "url": url
                }
                with open(META_PATH, "a") as mf:
                    mf.write(json.dumps(meta_entry) + "\n")

                os.remove(wav_path)
                f.write(f"DONE: {url}\n")
                print(f"Vaulted {info['id']} -> Shard {shard_idx}")

            except Exception as e:
                print(f"Error processing {url}: {e}")
                f.write(f"{url}\n")
        f.truncate()