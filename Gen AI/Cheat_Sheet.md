# Generative AI Mid-Term Cheat Sheet 🚀

This is a high-yield, quick-revision cheat sheet covering the core concepts, formulas, and differences across all 4 modules.

---

## 1. Mathematical Foundations & MLE

### Core Probability & Information Theory
* **Bayes' Theorem:** $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$
  * **Prior:** $P(Y)$ (Belief before evidence)
  * **Likelihood:** $P(X|Y)$ (How likely evidence is given the class)
  * **Posterior:** $P(Y|X)$ (Updated belief)
* **Shannon Entropy:** $H(P) = -\sum P(x) \log P(x)$. Measures average uncertainty.
* **KL-Divergence ($D_{KL}$):** $D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$
  * **Properties:** Non-negative ($\ge 0$), **Asymmetric** ($D_{KL}(P||Q) \neq D_{KL}(Q||P)$).
  * **Forward KL ($P||Q$):** Mean-seeking (spreads out).
  * **Reverse KL ($Q||P$):** Mode-seeking (locks onto a peak). VAEs use this!
* **Cross-Entropy:** $H(P, Q) = H(P) + D_{KL}(P || Q)$. Minimizing cross-entropy minimizes KL-Divergence.

### Maximum Likelihood Estimation (MLE)
* **Goal:** Find parameters $\theta$ that maximize the probability of observed data.
* **Objective:** $\theta^* = \arg\max_\theta \sum_{i=1}^m \log P_\theta(x^{(i)})$
* **Equivalence:** Maximizing log-likelihood $\iff$ Minimizing Forward KL-Divergence $D_{KL}(P_{data} || P_\theta)$.

---

## 2. Latent Variable Models & VAEs

### AE vs. VAE
* **Autoencoder (AE):** Maps input to a single point. Good for compression, bad for generation (gaps in latent space).
* **Variational AE (VAE):** Maps input to a *distribution* (mean & variance). Forces a continuous latent space $\mathcal{N}(0, I)$ allowing sampling and generation.

### Variational Inference & ELBO
* The true posterior $p(z|x)$ is intractable. We approximate it with the encoder $q(z|x)$.
* **ELBO (Evidence Lower Bound):** 
  $$\log p(x) \ge \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) || p(z))}_{\text{Regularization}}$$
* Minimizing the KL divergence between our approximation and the true posterior ($D_{KL}(q||p)$) **maximizes the ELBO**.

### The Reparameterization Trick
* **Why?** You cannot backpropagate gradients through a random sampling node.
* **How?** Move randomness to an independent variable $\epsilon \sim \mathcal{N}(0, 1)$.
* **Formula:** $z = \mu + \sigma \odot \epsilon$. Now $\mu$ and $\sigma$ are differentiable.

---

## 3. Generative Adversarial Networks (GANs)

### Architecture & Minimax Game
* **Generator ($G$):** Tries to create fake data from noise $z$ to fool $D$.
* **Discriminator ($D$):** Binary classifier deciding if data is real ($x$) or fake ($G(z)$).
* **Objective:** $\min_G \max_D V(G, D) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log (1 - D(G(z)))]$
* **JSD Connection:** If $D$ is optimal, training $G$ is mathematically equivalent to minimizing the **Jensen-Shannon Divergence (JSD)**.

### Challenges
1. **Mode Collapse:** $G$ produces only a few distinct outputs (or just one) that fool $D$, losing dataset diversity.
2. **Unstable Optimization:** Since it's a game, gradients can vanish/explode. No clear global minimum, just a Nash Equilibrium.

### Evaluation Metrics
* **Inception Score (IS):** Measures Sharpness (confident classification) and Diversity (triggers all classes evenly). **Higher is better.**
* **Frechet Inception Distance (FID):** Compares feature distribution of real vs. fake images. **Lower is better** (0 = perfect match).

### GAN Variants
* **cGAN:** Adds condition/label $y$ to guide generation (e.g., text-to-image).
* **WGAN:** Uses Wasserstein distance instead of JSD. Solves vanishing gradients and stabilizes training.
* **CycleGAN:** For unpaired image-to-image translation. Uses **Cycle Consistency Loss** ($x \to Y \to X \approx x$).
* **BiGAN:** Adds an Encoder to allow inferring latent vector $z$ from real image $x$.

---

## 4. Diffusion Models

### The Pipeline
* **Forward (Noising):** Fixed Markov chain that slowly adds Gaussian noise to an image over $T$ steps until it's pure noise.
* **Reverse (Denoising):** Neural Network (U-Net) trained to step-by-step remove noise from pure Gaussian noise to create an image.

### Mathematical Formulation
* **Forward Step:** $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$
* **Variance Schedule ($\beta_t$):** Controls how much noise is added. Kept small so the reverse step can be modeled as a Gaussian.
* **Why $x_T$ becomes pure noise:** Let $\alpha_t = 1-\beta_t$. The direct sample is $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$. As $t \to T$, $\bar{\alpha}_T \to 0$, eliminating $x_0$ and leaving only noise $\epsilon \sim \mathcal{N}(0, I)$.
* **Relation to VAE:** Diffusion is essentially a hierarchical VAE with $T$ layers. It is trained by maximizing the ELBO.

### Challenges & Solutions
* **Slow Sampling:** Standard diffusion needs 1,000 passes through the U-Net. **DDIM** solves this by skipping steps without retraining.
* **High-Res / Compute Heavy:** Operating on pixels is expensive. **Latent Diffusion (Stable Diffusion)** solves this by doing the diffusion process inside the tiny, compressed latent space of a VAE.
* **Conditional Generation:** **Classifier-Free Guidance (CFG)** pushes the generated image heavily toward the text prompt by extrapolating away from an unconditional prediction.
