- `mkdir data;mkdir checkpoints; touch ./data/urls.txt`
- `python3 -m venv .venv; source ./.venv/bin/activate`
- `pip install jax optax flax numpy scipy`
- `brew install python-tk`
- add youtube urls to `/data/urls.txt`; do not modify "DONE" entries.
- `brew install caffeinate`
- `python3 model.py` and `python3 inference.py` can be run concurrently
  
---

<img width="1636" height="905" alt="image" src="https://github.com/user-attachments/assets/0914d77f-6669-4502-b6be-bb135e75fd33" />

* **Red Squares (The Consensus):** In the left chart, most queries (rows 9–12 and 16–18) anchor heavily to **Key 6**. Because they share the exact same context provider, their resulting features merge into identical copycats, generating the solid red block-diagonal clusters on the right.
* **Blue Dots (The Divergences):** The right chart flashes sharp blue squares at the intersections of **Tokens 1, 5, and 14**. In the left chart, these queries completely ignore the popular Key 6 and shoot backward to lock onto **Key 1**. Because this behavior opposes the global mean, centering the data exposes it as a strong negative correlation.
* **The Blue Crosshair (Token 8):** A light-blue horizontal and vertical stripe cuts directly through **Index 8**. In the left chart, Row 8 ignores single-source peaks and instead smears its attention broadly across Keys 9 through 19. This unique, distributed footprint decouples it entirely from the rest of the sequence.
