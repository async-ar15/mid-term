# Gen AI Mid-Term Study Plan

Based on the syllabus lectures and the Past Year Questions (PYQs), here is a detailed, structured study plan to help you prepare effectively for your Generative AI (CS441) mid-term exam.

## 🎯 Exam Overview
* **Time:** 2 Hours
* **Max Marks:** 30
* **Structure:** 3 Main Questions (10 Marks each), with internal choices (OR) for every question.
* **Core Themes:** The exam heavily focuses on the mathematical foundations, training pipelines, and specific challenges of the "Big Three" generative models: **GANs, VAEs, and Diffusion Models**.

---

## 📚 Module-wise Study Plan

### 1. Mathematical Foundations & MLE (Lectures 3, 4, 6)
**Priority: Medium** (Crucial for understanding VAEs and GANs)
* **Topics to Cover:**
  * Probability theory (Bayes theorem, joint/conditional probabilities).
  * Information Theory: Shannon Entropy, **KL-Divergence** (definition, asymmetry, why it's used).
  * **Maximum Likelihood Estimation (MLE):** How learning a generative model is framed as density estimation.
  * Monte Carlo Estimation and the bias-variance tradeoff.
* **Study Tip:** Ensure you understand the math behind KL-divergence as it forms the basis of the VAE loss function and Jenson-Shannon Divergence in GANs.

### 2. Latent Variable Models & VAEs (Lectures 7, 8)
**Priority: HIGH (Guaranteed 10-Mark Question)**
* **Topics to Cover:**
  * **Autoencoders vs. Variational Autoencoders:** Why VAEs are generative and vanilla AEs are not (continuous latent space, sampling capability).
  * **Variational Inference:** Understand the Evidence Lower Bound (ELBO). Be able to prove how minimizing $KL(q(z|x) || p(z))$ maximizes ELBO.
  * Why $KL(q(z|x) || p(z))$ is used instead of $KL(p(z) || q(z|x))$.
  * **The Reparameterization Trick:** Why it is needed (to backpropagate through a random sampling process) and the exact formula: $z = \mu + \sigma \odot \epsilon$.
* **Actionable Practice:**
  * **Numerical:** Practice calculating the latent vector $z$ given $\mu$, $\sigma$, and a random noise sample $\epsilon$ from $\mathcal{N}(0, 1)$. (e.g., Q2 from PYQ).
  * **Conceptual:** Draw and explain the Variational Inference ellipse diagram, showing true posterior vs. approximate distribution.

### 3. Generative Adversarial Networks - GANs (Lectures 9, 10, 11)
**Priority: HIGH (Guaranteed 10-Mark Question)**
* **Topics to Cover:**
  * **Adversarial Training:** The core idea of generator vs. discriminator and the min-max optimization problem.
  * **Loss Functions:** Jenson-Shannon Divergence (JSD) and Cross-Entropy loss for discriminator/generator.
  * **Training Challenges:** 
    * **Mode Collapse:** What it is, why it happens, and how to fix it.
    * **Non-convergence / Unstable optimization.**
  * **Evaluation Metrics:** Inception Score (IS), Frechet Inception Distance (FID), Kernel Inception Distance (KID). Understand how they measure sharpness and diversity.
* **Actionable Practice:**
  * **Numerical:** Practice computing Discriminator Loss ($L_D$) and Generator Loss ($L_G$) given probabilities of real and generated samples. Understand what the values imply about the performance of D and G. (e.g., Q1 from PYQ).
  * **Variants:** Prepare detailed notes on at least two GAN variants that solve mode collapse/convergence (e.g., WGAN, CGAN).

### 4. Diffusion Models (Implied from Syllabus / PYQ)
**Priority: HIGH (Guaranteed 10-Mark Question)**
* **Topics to Cover:**
  * **Training & Sampling Pipeline:** Forward (noising) process and Reverse (denoising) process.
  * **Mathematical Formulation:** 
    * Forward process as a Markov chain: $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$.
    * The significance of the variance schedule $\beta_t$.
    * Why $x_T$ converges to pure Gaussian noise.
    * Generative process $p_\theta(x_{t-1}|x_t)$ and its relation to Variational Inference/VAEs.
    * Stochastic Differential Equations (SDE) representation of diffusion.
  * **Challenges & Solutions:** 
    * Accelerating slow sampling.
    * Adapting for high-resolution conditional generation (e.g., classifier-free guidance, latent diffusion).
* **Study Tip:** Make sure you can write down the exact equations for both the forward and reverse processes.

---

## 📝 Practice Checklist (Based on PYQ)
Make sure you can confidently answer the following before the exam:

- [ ] Calculate $L_D$ and $L_G$ for a batch of GAN outputs.
- [ ] Explain mode-collapse in GANs with a numerical/visual example.
- [ ] Compute a latent vector $z$ using the VAE reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
- [ ] Derive the ELBO and explain why minimizing KL divergence maximizes ELBO.
- [ ] Write the Markov process equation for a DDPM's forward step and explain $\beta_t$.
- [ ] Explain 2 strategies to speed up Diffusion model sampling.

## 🚀 Suggested 3-Day Revision Timeline
* **Day 1: Math & VAEs**
  * Revise Probability, KL Divergence, and MLE.
  * Master VAEs: Architecture, Reparameterization Trick (do numericals), ELBO derivation.
* **Day 2: GANs**
  * Master GAN architecture, Min-Max loss, and numerical loss computation.
  * Study Mode Collapse, non-convergence, and 2 GAN variants (e.g., WGAN).
  * Briefly review Evaluation metrics (IS, FID).
* **Day 3: Diffusion Models & Mock Test**
  * Focus heavily on the math of the forward/reverse diffusion process.
  * Study sampling acceleration and conditional generation.
  * Re-solve the entire PYQ paper blindly in 2 hours.
