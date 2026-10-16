### deploy

* `./setup-cdk.sh && aws configure`
* `cdk bootstrap && cdk deploy --profile` (Dashboard: `http://localhost:8000/dashboard.png`)

---

### Model

$$f(x; \theta) = W_{\text{up2}} \cdot \sigma(W_{\text{up1}} \cdot \text{LN}(h + \text{FF}(h)))$$

#### Symbols:
* **$x$**: Input tensor $(B, T, C)$ (Batch, Time, Channels).
* **$\theta$**: Flattened trainable parameter vector.
* **$f(x; \theta)$**: Dual-stem audio output (vocals + instrumentals).
* **$h$**: Hidden representation: $h = \text{Attention}(x) + x$.
* **$\text{FF}(h)$**: Feed-Forward feature extraction.
* **$\text{LN}$**: Layer Normalization.
* **$\sigma$**: GELU activation.
* **$W_{\text{up1}}, W_{\text{up2}}$**: Up-projection weight matrices.

---

### `model.py` + `meta.py`

$$\text{Inner Loop}: \quad \theta^{(k+1)} = \theta^{(k)} - \eta \cdot \nabla_\theta \mathcal{L}_{\text{diff}}(\theta^{(k)})$$

$$\text{Streaming Jacobian}: \quad \xi_t = \text{pad}\left( \frac{1}{\vert{}C_i\vert{}} \sum_{r \in C_i} J_r, 1024 \right) \oplus [\delta_t, L_t]$$

$$\text{Meta-Preconditioned Update}: \quad \theta_{t+1} = \text{CauchyQuant}\left( \theta_t - \eta \cdot \left[ P(\xi_t; \phi) \odot \nabla L_t \right] \right)$$

$$\text{Manifold Correction}: \quad W^{(l)}_{t+1} = W^{(l)}_t \cdot \text{exptm}\left(-\epsilon_{\text{lie}} \cdot \text{skew}\left(\nabla_{W^{(l)}} L_t\right)\right)$$

#### Symbols:
* **$\mathcal{L}_{\text{diff}}$**: Diffusion-denoising MSE loss.
* **$C_i, J_r$**: Jacobian chunk indices and row slices.
* **$\delta_t$**: Normalized parameter drift.
* **$\text{exptm}, \text{skew}$**: Matrix exponential and skew-symmetric projection ($0.5(G - G^\top)$).

---

### Gradient Checkpointing, Quantization & Cauchy Merging

**Context:** Uniform quantization clips high-magnitude outliers, causing gradient stagnation in attention weights. Cauchy quantization maps deltas via the Cauchy CDF to preserve heavy-tailed outlier updates.

$$\text{Gradient Checkpointing}: \quad h_l = f_l\left(f_{l-1}\left(\dots \text{checkpoint}(h_k) \dots \right)\right) \quad \text{s.t.} \quad \text{Memory}_{\text{act}} = \mathcal{O}\left(\frac{L}{\sqrt{K}}\right)$$

$$\text{1. Cauchy CDF Mapping}: \quad F(\delta) = \frac{1}{\pi} \arctan\left(\frac{\delta}{\gamma}\right) + 0.5$$

$$\text{2. Uniform Quantization}: \quad \tilde{F} = \text{clip}\left( \frac{\text{round}(F(\delta) \cdot M)}{M}, \frac{1}{2M}, 1 - \frac{1}{2M} \right)$$

$$\text{3. Inverse Reconstruction}: \quad \delta_{\text{quant}} = \gamma \cdot \tan\left(\pi \left(\tilde{F} - 0.5\right)\right)$$

$$\text{4. Parameter Merge}: \quad \theta_{\text{merged}} = \theta_{\text{base}} + \delta_{\text{quant}}$$

#### Symbols:
* **$h_l, h_k$**: Intermediate activation tensors at layers $l$ and checkpoint boundary $k$.
* **$L$**: Total transformer layer depth.
* **$\gamma$**: Cauchy scale parameter ($0.1$).
* **$M$**: Number of discrete quantization bins ($16$).
* **$\theta_{\text{base}}, \theta_{\text{adapt}}$**: Base and adapted parameters.

---

### Neural Tangent Kernel (NTK)

$$\Theta_t(x, x') = \sum_{k=1}^{P} \frac{\partial f(x; \theta_t)}{\partial \theta_k} \otimes \frac{\partial f(x'; \theta_t)}{\partial \theta_k}$$

#### Symbols:
* **$\Theta_t(x, x')$**: Structural input similarity kernel.
* **$P$**: Total scalar parameters.
* **$\theta_k$**: Individual scalar parameter weight.
* **$\frac{\partial f(x; \theta_t)}{\partial \theta_k}$**: Gradient Jacobian with respect to parameter $\theta_k$.
* **$\otimes$**: Kronecker tensor outer product.

---

### `meta.py` (Spectral Preconditioner)

$$\Delta \theta_t = P(\xi_t; \phi) \odot \nabla L_t$$

$$P = \sigma\left( \mathcal{M}_{\phi}(\xi_t) \right)$$

$$\xi_t = \text{NTK}_t \oplus \delta_t$$

$$\delta_t = \frac{\Vert{}\theta_t - \theta_{t-1}\Vert{}_2}{\Vert{}\theta_t\Vert{}_2 + \epsilon}$$

$$\sigma(x) = 2.0 \cdot \left( \frac{1}{1 + e^{-x}} \right)$$

#### Symbols:
* **$\Delta \theta_t$**: Modulated parameter update.
* **$P$**: Adaptive spectral preconditioner scaling vector.
* **$\xi_t$**: Concatenated curvature and drift input feature vector.
* **$\phi$**: Trainable Meta-MLP parameters.
* **$\nabla L_t$**: Raw loss gradient.
* **$\odot$**: Hadamard element-wise product.
* **$\mathcal{M}_{\phi}$**: 2-layer GELU Meta-MLP.
* **$\text{NTK}_t$**: Flattened 1024-dim local curvature feature.
* **$\epsilon$**: Stability constant ($10^{-6}$).
* **$\oplus$**: Concatenation operator.

---

### Scaling Laws & Generalization Bounds

* **Proposition**: In the infinite-width limit, standard scaling laws collapse to NTK spectral bounds, mapping 1s audio diffusion inversion to kernel ridge regression governed by $E_{1\text{s}} \approx 0.1 \cdot \kappa(\Theta^\infty) \ln(1/\epsilon)$.
* **Definition**: $D_{\text{samples}} \ge \kappa(\Theta^\infty) \cdot \ln(1/\epsilon)$ defines the minimum distinct data volume required to span attention NTK degrees of freedom.

$$\hat{f} = \arg\min_{f \in \mathcal{H}_{\Theta^\infty}} \frac{1}{N_{\text{songs}}} \sum_{i=1}^{N_{\text{songs}}} \left( y_i - f(x_i) \right)^2 + \lambda_{\text{reg}} \Vert{}f\Vert{}_{\mathcal{H}_{\Theta^\infty}}^2 \quad \text{\scriptsize (kernel ridge regression)}$$

$$\Theta^\infty(x, x') = \sum_{j=1}^{\infty} \mu_j \phi_j(x) \phi_j(x') \quad \text{\scriptsize (spectral expansion)}$$

$$d_{\text{eff}}(\lambda_{\text{reg}}) = \text{Tr}\left( \Theta^\infty (\Theta^\infty + \lambda_{\text{reg}} I)^{-1} \right) \quad \text{\scriptsize (effective dimensionality)}$$

$$N_{\text{songs}} \ge \frac{\text{Tr}(\Theta^\infty)}{\lambda_{\text{reg}}} \cdot \ln\left(\frac{1}{\epsilon}\right) \approx \mathcal{O}\left(\kappa(\Theta^\infty) \cdot d_{\text{attn}}\right) \quad \text{\scriptsize (generalization bound)}$$

$\blacksquare$

---

### Symbols & Hyper-Parameters

* **$f_s = 16\text{ kHz}$**: Uniform audio sampling baseline.
* **128-sample frames**: Temporal window $\Delta t = 8\text{ ms}$.
* **$125\text{ tokens/s}$**: Hop frequency $f_{\text{hop}} = 125\text{ Hz}$.
* **$N_{\text{samples}} \ge 10^5$**: Minimum distinct compositional cardinality.
* **$\Theta^\infty$**: Attention NTK in the infinite-width limit.
* **$\mathcal{H}_{\Theta^\infty}$**: Induced RKHS.
* **$\lambda_{\text{reg}}$**: $L_2$ regularization parameter.
* **$\epsilon$**: Target reconstruction error ($10^{-4}$).
* **$\kappa(\Theta^\infty)$**: NTK condition number ($\in [10^2, 10^3]$).
* **$\mu_j, \phi_j$**: Eigenvalues and eigenfunctions.
* **$d_{\text{attn}}$**: Internal attention feature dimension ($4096$, $16$ heads, $d_{\text{head}} = 256$).
* **a) FFNN iterations ($T_{\text{sec}}$)**: $I_{\text{iter}}(T_{\text{sec}}) \approx 92 \cdot T_{\text{sec}}$ ($921$ iterations for $10\text{s}$).
