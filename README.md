infinite-width CALM Attention Network ⚠️

---

### deploy

* `./setup-cdk.sh && aws configure`
* `cdk bootstrap && cdk deploy --profile` (Dashboard: `http://localhost:8000/dashboard.png`)

---

### Model

$$f(x; \theta) = W_{\text{up2}} \cdot \sigma(W_{\text{up1}} \cdot \text{LN}(h + \text{FF}(h)))$$


* **$x$**: Input $(B, T, C)$ audio sequence.
* **$h$**: Hidden representation ($\text{Attention}(x) + x$).
* **$\theta, W_{\text{up1/2}}$**: Parameters and dual-stem output weights.

---

### Neural Tangent Kernel (NTK) & Preconditioner

$$\Theta_t(x, x') = \sum_{k=1}^{P} \frac{\partial f(x; \theta_t)}{\partial \theta_k} \otimes \frac{\partial f(x'; \theta_t)}{\partial \theta_k}$$


*Parameter-space Jacobian metric defining local network curvature.*

$$\Delta \theta_t = \sigma(\mathcal{M}_{\phi}(\text{NTK}_t \oplus \delta_t)) \odot \nabla L_t$$


*Adaptive meta-gradient update modulated by curvature and parameter drift.*

* **$\delta_t$**: Symmetry drift ($\frac{\Vert\theta_t - \theta_{t-1}\Vert_2}{\Vert\theta_t\Vert_2 + \epsilon}$).

---

$$E_{1\vert{}s} \approx 0.1 \cdot \kappa(\Theta^\infty) \ln(1/\epsilon)$$


*Reconstruction error bound governed by kernel condition number and precision.*

$$D_{\text{samples}} \ge \kappa(\Theta^\infty) \cdot \ln(1/\epsilon) \quad (N_{\text{samples}} \ge 10^5)$$


*Minimum distinct data cardinality required to span attention degrees of freedom.*

* **$f_s$ / Stride:** $16$ kHz baseline; $128$-sample frames ($\Delta t = 8$ ms); $125$ tokens/s stride.

$$\hat{f} = \arg\min_{f \in \mathcal{H}_{\Theta^\infty}} \frac{1}{N_{\text{samples}}} \sum (y_i - f(x_i))^2 + \lambda_{\text{reg}} \Vert{}f\Vert{}_{\mathcal{H}_{\Theta^\infty}}^2$$


*Infinite-width limit formulated as kernel ridge regression.*

$$\Theta^\infty(x, x') = \sum_{j=1}^{\infty} \mu_j \phi_j(x) \phi_j(x')$$


*Mercer's spectral expansion of the infinite-width attention kernel.*

$$N_{\text{samples}} \ge \frac{\text{Tr}(\Theta^\infty)}{\lambda_{\text{reg}}} \cdot \ln\left(\frac{1}{\epsilon}\right) \approx \mathcal{O}\left(\kappa(\Theta^\infty) \cdot d_{\text{attn}}\right)$$



---

### Config (8GB VRAM)

* **$d_{\text{attn}}$**: $4096$ ($16$ heads, $d_{\text{head}} = 256$).
* **$I_{\text{iter}}$**: $\approx 92 \cdot T_{\text{sec}}$.
