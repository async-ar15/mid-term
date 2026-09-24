# Generative AI Mid-Term: PYQ Complete Solutions

This document contains detailed, exam-style solutions for the Past Year Questions (PYQs) provided in `pyq.md`. Both options for every question have been solved so you have a complete reference.

---

## Question 1: Generative Adversarial Networks (GANs)

### First Option
**1. Core Idea of Adversarial Training & Architecture:**
Generative Adversarial Networks (GANs) consist of two neural networks, a Generator ($G$) and a Discriminator ($D$), trained simultaneously in a competitive zero-sum game.
* **Generator ($G$):** Takes a random noise vector $z \sim p_z$ as input and learns to map it to the data space, producing fake samples $G(z)$ that mimic the true data distribution $p_{data}$. Its objective is to maximize the probability of $D$ making a mistake.
* **Discriminator ($D$):** A binary classifier that takes an input sample $x$ and outputs a probability $D(x) \in [0, 1]$ indicating whether the sample is real (from $p_{data}$) or fake (from $G$). Its objective is to accurately distinguish between real and fake samples.

**2. Min-Max Optimization Problem:**
The training is expressed as a minimax game with the value function $V(G,D)$:
$$
\min_G \max_D V(G, D) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$
**Justification:** 
* The Discriminator wants to maximize $V(G,D)$. It pushes $D(x) \to 1$ (making $\log D(x) \to 0$) and $D(G(z)) \to 0$ (making $\log(1 - D(G(z))) \to 0$).
* The Generator wants to minimize $V(G,D)$. It has no control over the first term, but it wants $D(G(z)) \to 1$ (fooling the discriminator), which drives $\log(1 - D(G(z))) \to -\infty$. 
Thus, $D$ maximizes the objective while $G$ minimizes it.

**3. Numerical Solution:**
Given batch of size $N=4$:
Real: $D(x) = [0.9, 0.8, 0.7, 0.95]$
Fake: $D(G(z)) = [0.2, 0.3, 0.1, 0.4]$

**A. Compute Discriminator Loss ($L_D$):**
Using binary cross entropy (where $D$ wants to minimize $-V(G,D)$):
$L_D = -\frac{1}{N} \sum \ln D(x) - \frac{1}{N} \sum \ln(1 - D(G(z)))$
Real part: $-\frac{1}{4} [\ln(0.9) + \ln(0.8) + \ln(0.7) + \ln(0.95)] = -\frac{1}{4} [-0.105 - 0.223 - 0.357 - 0.051] = 0.184$
Fake part: $-\frac{1}{4} [\ln(1-0.2) + \ln(1-0.3) + \ln(1-0.1) + \ln(1-0.4)]$
$=-\frac{1}{4} [\ln(0.8) + \ln(0.7) + \ln(0.9) + \ln(0.6)] = -\frac{1}{4} [-0.223 - 0.357 - 0.105 - 0.511] = 0.299$
$L_D = 0.184 + 0.299 = 0.483$

**B. Compute Generator Loss ($L_G$):**
Using the minimax formulation:
$L_G = \frac{1}{N} \sum \ln(1 - D(G(z)))$
$L_G = \frac{1}{4} [\ln(0.8) + \ln(0.7) + \ln(0.9) + \ln(0.6)] = -0.299$

**C. Interpretation:**
* The Discriminator loss is quite low ($0.483$), indicating it is highly confident and successful at distinguishing real images (giving scores $\ge 0.7$) from fake images (giving scores $\le 0.4$).
* The Generator is currently failing to fool the discriminator. Because $D(G(z))$ values are far from 1, the gradients for $G$ might start vanishing if the strict minimax loss is used, making training difficult.

---

### OR Option (Challenges & Variants)
**1. Major Challenges:**
* **Non-convergence / Unstable Optimization:** GANs search for a Nash Equilibrium rather than a global minimum. Because the loss landscape changes dynamically for one network as the other updates, gradients can vanish or explode. If $D$ becomes too perfect too quickly, $D(G(z)) \to 0$, and the gradient for the generator approaches 0, meaning $G$ stops learning entirely.
* **Mode Collapse:** The generator discovers a small number of samples (modes) that consistently fool the discriminator and stops exploring the rest of the data distribution. 
  * *Numerical Example:* Suppose the training data consists of digits 0-9 evenly distributed. A mode-collapsed generator might output only perfect-looking '7's (100% of the time). Since the '7's look real, $D(G(z)) = 0.99$. The discriminator is fooled, so the generator never learns to generate '0's or '1's.

**2. Two GAN Variants to Overcome Challenges:**
* **Wasserstein GAN (WGAN):** Uses Earth Mover's Distance instead of Jensen-Shannon Divergence. It replaces the discriminator with a "Critic" that outputs a real-valued score instead of a probability $[0, 1]$, and enforces 1-Lipschitz continuity. This solves vanishing gradients and stabilizes training, heavily reducing mode collapse.
* **Conditional GAN (cGAN):** Feeds class labels (e.g., digit labels 0-9) to both $G$ and $D$. This forces the generator to map specific regions of the latent space $z$ to specific classes, heavily mitigating mode collapse by ensuring the generator must produce all classes.

---

## Question 2: Variational Autoencoders (VAEs)

### First Option
**1. VAEs vs Vanilla Autoencoders:**
Vanilla Autoencoders (AEs) are designed solely for data compression. They map inputs deterministically to single points in the latent space. Because the latent space is unregularized and has gaps, sampling a random point $z$ will likely produce meaningless noise.
Variational Autoencoders (VAEs) are generative because they map inputs to *probability distributions* (mean and variance). The entire latent space is heavily regularized to follow a prior distribution $\mathcal{N}(0, I)$. Thus, any randomly sampled point from this continuous space will decode into a valid, realistic data sample.

**2. The Reparameterization Trick:**
* **What & How:** Instead of sampling $z$ directly from a dynamic distribution $\mathcal{N}(\mu, \sigma^2)$, we sample noise $\epsilon$ from a fixed standard normal distribution $\mathcal{N}(0, 1)$. We then compute $z$ deterministically: $z = \mu + \sigma \odot \epsilon$.
* **Why:** In neural networks, we compute gradients via backpropagation. However, the backpropagation algorithm cannot flow through a stochastic (random) sampling node. By separating the randomness into an independent variable $\epsilon$, the network operations involving $\mu$ and $\sigma$ become purely differentiable (just addition and multiplication), allowing the encoder weights to be updated.

**3. Numerical Computation:**
Given $\mu = [0.5, -1.0]$, $\sigma = [0.2, 0.5]$, $\epsilon = [0.1, -0.3]$.
Using $z = \mu + \sigma \odot \epsilon$:
$z_1 = 0.5 + (0.2 \times 0.1) = 0.5 + 0.02 = 0.52$
$z_2 = -1.0 + (0.5 \times -0.3) = -1.0 - 0.15 = -1.15$
**Result:** Latent vector $z = [0.52, -1.15]$.

---

### OR Option (Variational Inference & KL)
**Why Reverse KL ($KL(q || p)$) instead of Forward KL ($KL(p || q)$)?**
Forward KL requires taking an expectation with respect to the true posterior $p(z|x)$. However, $p(z|x)$ is precisely the analytically intractable distribution we cannot compute! Reverse KL takes the expectation over $q(z|x)$, which is our own encoder neural network that we can easily evaluate and sample from. Additionally, Reverse KL is "mode-seeking," fitting tightly around high-probability regions of the true distribution.

**Variational Inference (VI) Illustration Answers:**
* **A. Why true posterior lies outside:** The true posterior $p(z|x)$ represents the exact latent factors for a complex dataset and is highly complex, multimodal, and analytically intractable. The variational family $q(z;v)$ is intentionally restricted to a simpler, computationally tractable distribution class (like a Gaussian). Therefore, the highly complex true distribution naturally falls outside the limited expressivity of the simpler Gaussian family.
* **B. Equivalence to ELBO:** Using Bayes' theorem, we know:
  $KL(q(z;v) || p(z|x)) = \mathbb{E}_q [\log q(z;v) - \log p(z|x)]$
  $= \mathbb{E}_q [\log q(z;v) - \log p(x,z) + \log p(x)]$
  $= -\text{ELBO} + \log p(x)$
  Since the log-evidence $\log p(x)$ is a fixed constant, any decrease in the KL divergence term must correspond to an exactly equal increase in the ELBO term. Thus, minimizing $KL$ maximizes ELBO.
* **C. The path from $v^{init}$ to $v^*$:** This path represents the gradient-based optimization process (training the encoder). $v^{init}$ represents the randomly initialized weights of the encoder. $v^*$ represents the optimal weights where the approximate distribution $q$ is as close as mathematically possible to the true posterior $p(z|x)$, yielding the highest possible ELBO for that specific model architecture.

---

## Question 3: Diffusion Models

### First Option
**A. $\beta_t$ and its significance:**
$\beta_t \in (0, 1)$ is the variance schedule for the diffusion model. It dictates the exact fraction of variance (noise) injected at time step $t$. Its significance lies in ensuring that the information destruction is gradual; by keeping $\beta_t$ sufficiently small at each step, the reverse denoising transition $p_\theta(x_{t-1}|x_t)$ can be mathematically approximated as a Gaussian distribution.

**B. Markov Process and $x_T$ Convergence:**
The forward process is a Markov chain because the state $x_t$ depends *only* on the immediately preceding state $x_{t-1}$, defined by $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$. 
Using the reparameterization trick, we define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, allowing us to sample $x_t$ directly from $x_0$:
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
$$
As $t \to T$, because each $\alpha_i < 1$, the infinite product $\bar{\alpha}_T \to 0$. Therefore, the mean term $\sqrt{\bar{\alpha}_T}x_0 \to 0$ and the variance term $(1 - \bar{\alpha}_T) \to 1$. The state $x_T$ perfectly converges to pure Gaussian noise $\mathcal{N}(0, \mathbf{I})$, entirely destroying $x_0$.

**C. Generative Process $p_\theta$ and VAE relation:**
The generative process reverses the Markov chain, using a neural network to estimate the parameters:
$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
**Relation to VAE:** A diffusion model can be viewed as a hierarchical, very deep VAE with $T$ latent layers. The forward noising process acts as a fixed, parameter-free encoder $q$, and the reverse denoising process acts as a highly expressive decoder $p_\theta$. Like a VAE, diffusion models are trained by maximizing the Evidence Lower Bound (ELBO) over these $T$ layers.

**D. SDE Representation:**
As $T \to \infty$, the discrete steps become a continuous time variable $t \in [0, 1]$.
* **Forward SDE:** $dx = f(x, t)dt + g(t)dw$ (where $w$ is a standard Wiener process).
* **Reverse SDE:** $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$ (where $\nabla_x \log p_t(x)$ is the score function learned by the network).

---

### OR Option (Challenges & Strategies)
**A. Accelerating Sampling Strategies:**
The standard Denoising Diffusion Probabilistic Model (DDPM) is slow because it strictly requires traversing all $T$ (e.g., 1000) steps sequentially due to the Markov assumption.
* **Denoising Diffusion Implicit Models (DDIM):** Formulates the forward process as non-Markovian while maintaining the exact same marginal distributions $q(x_t|x_0)$. Because the generative process becomes deterministic, the model can safely skip intermediate steps during inference. For example, a model trained on 1000 steps can generate a high-quality image in just 50 steps without requiring retraining.
* **Probability Flow ODEs:** The reverse SDE can be cast as an equivalent Ordinary Differential Equation (ODE). Advanced numerical ODE solvers (like Runge-Kutta methods) can be applied to traverse the integration path in far fewer steps with high precision.

**B. High-Resolution Conditional Generation:**
* **High-Resolution (Latent Diffusion):** Running a U-Net on a massive pixel grid (e.g., 1024x1024) for 1000 steps is computationally intractable. Latent Diffusion Models (LDMs) solve this by using a pre-trained VAE encoder to compress the image into a small, dense latent space. The entire diffusion (noising/denoising) process is executed in this cheap latent space. Finally, a VAE decoder projects the generated latent vector back into high-resolution pixel space.
* **Conditional Generation (Classifier-Free Guidance):** To strongly align generation with a text condition $c$, the model is trained jointly with the condition $\epsilon_\theta(x_t, c)$ and without the condition $\epsilon_\theta(x_t, \emptyset)$ (by dropping the text prompt 10% of the time during training). During sampling, the final noise prediction is extrapolated away from the unconditional prediction: 
  $$
  \hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
  $$
  This pushes the generation heavily toward the text prompt without needing an external classifier network.

---
---

## Exam Paper Solutions: Mid-Semester 2025-2026 (Rajiv Gandhi Institute of Petroleum Technology)

### Q1: Generative Adversarial Networks (Main)
**1. Core Idea & Architecture:** 
Adversarial training involves two neural networks, a Generator ($G$) and a Discriminator ($D$), locked in a zero-sum minimax game. $G$'s objective is to take random noise and generate realistic fake data to fool $D$. $D$'s objective is to accurately classify data as either coming from the real dataset or the fake generated set.
**Min-Max Justification:** The loss is framed as a single objective function $V(D,G)$. The Discriminator tries to maximize it (assigning high probability to reals and low to fakes), while the Generator tries to minimize it (forcing the Discriminator to assign high probability to fakes).

**Numerical Computation:**
Given Reals $D(x) = [0.9, 0.8, 0.7, 0.95]$ and Fakes $D(G(z)) = [0.2, 0.3, 0.1, 0.4]$.
1. **Compute $L_D$:**
   $L_D = -\frac{1}{N} \sum \ln(D(x_i)) - \frac{1}{N} \sum \ln(1 - D(G(z_i)))$
   * Real term: $\frac{\ln(0.9) + \ln(0.8) + \ln(0.7) + \ln(0.95)}{4} = \frac{-0.105 - 0.223 - 0.357 - 0.051}{4} = -0.184$
   * Fake term: $1-D(G(z)) = [0.8, 0.7, 0.9, 0.6]$
     $\frac{\ln(0.8) + \ln(0.7) + \ln(0.9) + \ln(0.6)}{4} = \frac{-0.223 - 0.357 - 0.105 - 0.511}{4} = -0.299$
   * $L_D = -(-0.184) - (-0.299) = 0.184 + 0.299 = \textbf{0.483}$

2. **Compute $L_G$:**
   Using the standard non-saturating generator loss $L_G = -\frac{1}{N} \sum \ln(D(G(z_i)))$
   $L_G = -\frac{\ln(0.2) + \ln(0.3) + \ln(0.1) + \ln(0.4)}{4} = -\frac{-1.609 - 1.204 - 2.303 - 0.916}{4} = -\frac{-6.032}{4} = \textbf{1.508}$

3. **Interpretation:**
   The Discriminator loss ($0.483$) is very low, meaning it is successfully and confidently classifying reals from fakes. The Generator loss ($1.508$) is quite high. This implies the Discriminator is currently winning the minimax game easily, and the Generator is failing to fool it at this training step.

---

### Q1: Generative Adversarial Networks (OR Option)
* **Major Challenges:**
  * **Non-convergence:** The minimax game causes the loss landscape to constantly shift. If $G$ updates to exploit $D$'s weakness, $D$ immediately updates to close the loophole, creating an endless oscillating loop where gradients fluctuate and the model never settles into a stable Nash Equilibrium.
  * **Mode-collapse:** $G$ discovers a single specific "mode" (e.g., generating only pictures of the number '7') that consistently fools $D$. Instead of generating a diverse range of samples, $G$ collapses to outputting only this single mode to safely minimize its loss.
* **GAN Variants to Overcome Challenges:**
  1. **Wasserstein GAN (WGAN):** Uses Earth Mover's Distance instead of JS Divergence. It replaces the Discriminator with a "Critic" that uses weight clipping to ensure 1-Lipschitz continuity. This provides smooth, non-vanishing gradients everywhere, solving non-convergence.
  2. **Unrolled GANs or Mini-batch Discrimination:** Mini-batch discrimination allows the discriminator to look at an entire batch of generated images at once. If it detects zero variance (i.e. all images are the exact same collapsed mode), it flags them as fake, directly penalizing and mitigating mode-collapse.

---

### Q2: Variational Autoencoders (Main)
* **VAEs vs Vanilla Autoencoders:** Vanilla AEs map data to discrete, isolated deterministic points in the latent space. Decoding space between points yields garbage, so they cannot generate new data. VAEs enforce a continuous, probabilistic distribution $\mathcal{N}(\mu, \Sigma)$ across the latent space, keeping it densely packed and smooth so that sampling random points yields meaningful new data.
* **Reparameterization Trick:** Instead of sampling directly from the stochastic node $z \sim \mathcal{N}(\mu, \Sigma)$ (which blocks backpropagation because random sampling has no derivative), we sample independent noise $\epsilon \sim \mathcal{N}(0, I)$ and deterministically compute $z = \mu + \sigma \odot \epsilon$. This separates randomness from network weights, allowing gradients to flow backward through $\mu$ and $\sigma$.
* **Compute Latent Vector $z$:**
  Given $\mu = [0.5, -1.0]$, $\sigma = [0.2, 0.5]$, $\epsilon = [0.1, -0.3]$.
  $z_1 = \mu_1 + \sigma_1 \times \epsilon_1 = 0.5 + (0.2)(0.1) = 0.5 + 0.02 = 0.52$
  $z_2 = \mu_2 + \sigma_2 \times \epsilon_2 = -1.0 + (0.5)(-0.3) = -1.0 - 0.15 = -1.15$
  Latent vector **$z = [0.52, -1.15]$**.

---

### Q2: Variational Autoencoders (OR Option)
**Why $KL(q||p)$ instead of $KL(p||q)$?** We use $KL(q(z|x) || p(z|x))$ because $q$ is our defined, tractable approximate neural network distribution. The true posterior $p(z|x)$ is mathematically intractable to integrate directly, so we cannot compute expectations over it.

* **A. Why true posterior lies outside:** The true posterior $p(z|x)$ of complex image data is typically a highly convoluted, non-linear, and multi-modal shape. The variational family $q(z;v)$ is restricted to simple distributions (like a perfect Gaussian ellipse). A simple ellipse physically cannot twist to capture the complex true posterior, so the true posterior resides "outside" the representable family space.
* **B. Minimizing KL is Maximizing ELBO:**
  The log evidence formula is: $\log p(x) = \text{ELBO} + D_{KL}(q(z;v) || p(z|x))$.
  Because the total log evidence $\log p(x)$ is a fixed constant, any mathematical decrease in the KL divergence term forces an exactly equal increase in the ELBO term to balance the equation. Thus, minimizing KL directly maximizes ELBO.
* **C. Path and Roles:**
  * **Path:** The squiggly line represents the trajectory of Gradient Descent optimization during training.
  * **$v_{init}$:** The random starting weights of the neural network (Encoder) before training begins.
  * **$v*$:** The optimal set of trained weights where the approximate distribution $q$ is as close as mathematically possible to the true posterior $p$ (achieving the lowest KL divergence possible within that family).

---

### Q3: Diffusion Models (Main)
* **Training Pipeline:** Take a real image, iteratively add Gaussian noise over $T$ steps (Forward). Train a U-Net to predict the exact noise added at any step $t$, optimized using Mean Squared Error.
* **Sampling Pipeline:** Start with pure random Gaussian noise ($x_T$). Iteratively apply the trained U-Net over $T$ steps to predict and subtract out the noise, slowly denoising it into a pristine image.
* **A. $\beta_t$:** It is the predefined variance schedule. It dictates exactly how much noise is added at each specific timestep $t$ in the forward process.
* **B. Markov Process Equation:**
  $x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon$
  As $t \to T$, the cumulative signal $\bar{\alpha}_t$ approaches 0. In the equation $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, the image content term $\sqrt{\bar{\alpha}_t}x_0$ disappears completely, and the noise term coefficient becomes 1, leaving purely standard Gaussian noise $\epsilon$.
* **C. Generative Process $p_\theta$:**
  $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$.
  *Relation to VAE:* A diffusion model is mathematically equivalent to a Hierarchical VAE where the latent space has exactly the same dimensionality as the data, the encoders are fixed (no learnable parameters, just the $\beta$ schedule), and the decoders share weights across all $T$ layers.
* **D. SDE Representation:**
  * Forward SDE: $dx = f(x, t)dt + g(t)dw$
  * Reverse SDE: $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$

---

### Q3: Diffusion Models (OR Option)
* **A. Accelerating Sampling Strategies:**
  1. **DDIM (Denoising Diffusion Implicit Models):** Reformulates the forward process to be non-Markovian while keeping the same marginal distributions. This makes the generative process deterministic, allowing the model to safely skip hundreds of intermediate timesteps during generation (e.g., generating in 50 steps instead of 1000) without severe quality loss.
  2. **Probability Flow ODE Solvers:** The reverse SDE can be mapped to an equivalent ODE. High-order numerical solvers (like Runge-Kutta) can calculate the trajectory in much larger leaps than simple Euler stepping, vastly reducing the number of required sampling steps.
* **B. High-Resolution Conditional Generation:**
  * **Latent Diffusion Models (LDM) [Architecture]:** Running diffusion on massive pixel grids (1024x1024) is computationally intractable. An LDM uses a frozen VAE Encoder to compress the high-res image into a tiny latent map (e.g., 64x64). The entire 1000-step diffusion process runs extremely cheaply in this latent space, then the VAE Decoder upscales it back to high-res.
  * **Classifier-Free Guidance (CFG) [Mechanism]:** During training, the text condition $c$ is randomly dropped 10% of the time, so the U-Net learns both $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \emptyset)$. During sampling, generation is steered strongly toward the text prompt by extrapolating the noise prediction away from the unconditional prediction: 
  $\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$
