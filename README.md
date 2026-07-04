- `mkdir data;mkdir checkpoints; touch ./data/urls.txt`
- `python3 -m venv .venv; source ./.venv/bin/activate`
- `pip install jax optax flax numpy scipy`
- `brew install python-tk`
- add youtube urls to `/data/urls.txt`; do not modify "DONE" entries.
- `brew install caffeinate`
- `python3 model.py` and `python3 inference.py` can be run concurrently
  
---
<img width="1686" height="577" alt="image" src="https://github.com/user-attachments/assets/bdca5544-a2ff-4759-ae9d-ab93a17ad8fe" />

--- 
### Model Architecture

This autoregressive model takes an input sequence of audio tokens $x$ and projects them through a multi-head causal attention mechanism and feed-forward layer blocks to predict the next audio sequence:

$$f(x; \theta) = \mathbf{W}_{\text{up2}} \cdot \sigma \left( \mathbf{W}_{\text{up1}} \cdot \text{LN} \left( h + \text{FF}(h) \right) \right)$$

#### Symbol Key:
* **$x$**: Input audio token tensor of shape $(B, T, C)$, represents the sequential audio waveform.
* **$\theta$**: Flattened vector of all learnable weights and biases in the model at the current step.
* **$f(x; \theta)$**: The forward-pass output predicting the next chronological audio slice.
* **$h$**: The hidden representation vector emerging from the attention engine ($h = \text{Attention}(x) + x$).
* **$\text{FF}(h)$**: The feed-forward network block, defined as $\mathbf{W}_{\text{ff2}} \cdot \sigma(\mathbf{W}_{\text{ff1}} \cdot h)$.
* **$\text{LN}$**: Layer Normalization operator applied across the latent feature dimension.
* **$\sigma$**: The GELU (Gaussian Error Linear Unit) non-linear activation function.
* **$\mathbf{W}_{\text{up1}}, \mathbf{W}_{\text{up2}}$**: The weight matrices of the output projection dense layers (`up_proj_1` and `up_proj_2`).

---

### Empirical Neural Tangent Kernel (NTK)

The empirical NTK matrix $\Theta_t$ tracks how the network's output function generalizes and evolves at training step $t$ across two distinct input sequences, $x$ and $x'$, by measuring the geometric alignment of their parameter gradients:

$$\Theta_t(x, x') = \sum_{k=1}^{P} \frac{\partial f(x; \theta_t)}{\partial \theta_k} \otimes \frac{\partial f(x'; \theta_t)}{\partial \theta_k}$$

#### Symbol Key:
* **$\Theta_t(x, x')$**: The Neural Tangent Kernel value evaluating the structural similarity between inputs $x$ and $x'$ at step $t$.
* **$P$**: The total number of scalar parameters in the network.
* **$\theta_k$**: An individual scalar parameter weight within the active model parameters $\theta_t$.
* **$\frac{\partial f(x; \theta_t)}{\partial \theta_k}$**: The partial derivative (gradient Jacobian) of the model's prediction with respect to parameter $\theta_k$.
* **$\otimes$**: The Kronecker (or outer) product tensor operator, which matches the multidimensional output channel features of the audio tokens.
