- `mkdir data;mkdir checkpoints; touch ./data/urls.txt`
- `python3 -m venv .venv; source ./.venv/bin/activate`
- `pip install jax optax flax numpy scipy`
- `brew install python-tk`
- add youtube urls to `/data/urls.txt`; do not modify "DONE" entries.
- `brew install caffeinate`
- `python3 model.py` and `python3 inference.py` can be run concurrently
  
---
<img width="1686" height="577" alt="image" src="https://github.com/user-attachments/assets/bdca5544-a2ff-4759-ae9d-ab93a17ad8fe" />

* **Attention Matrix (H0–H7):** Visualizes per-head token routing patterns; enforces a lower-triangular causal mask where the y-axis (Query) map cannot look ahead into future x-axis (Key) tokens.
* **Feature Alignment Profile:** Displays the self-similarity matrix ($XX^T$) of mean-centered, normalized attention vectors to identify structural convergence, phase grouping, or representation collapse.
* **Parameter NTK Evolution:** Tracks the relative Frobenius norm deviation ($\Delta\Theta$) of the output projection weights against its initial state to track active feature learning vs. "lazy training" stagnation.
