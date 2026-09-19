# 1

The subject starts with a basic question: **what does a model learn from data** 

1. Discriminative model learns Pθ​(Y∣X)
2. Generative model learns the data distribution Pθ​(X) and can generate new samples
3. Conditional generative model learns Pθ​(X∣Y) allowing generation based on a given condition

The major deep-generative approaches covered include **VAEs, GANs, diffusion models and autoregressive models**

# 2

To understand these models, we first need
1. Probability
- Joint 
- conditional
- marginal
- sum/product rules
- Bayes' theorem

These concepts connect quantities such as p(x)p(x), p(z)p(z), p(x∣z)p(x|z) and p(z∣x)p(z|x), which appear repeatedly in generative modeling

2. Maximum Likelihood Estimation
- Where model parameters are chosen to maximize the probability of observed data.
- Log-likelihood makes this optimization easier
# 3

**Information theory** which gives us ways to measure uncertainty and differences between Distributions:
- Self-information 
- Entropy 
- Cross-entropy 
- KL divergence

KL divergence becomes especially important later in **variational inference and VAE**, while divergence measures also appear in **GAN mathematics**.

# 4

The major challenge is that real-world data is high-dimensional and complicated. Many meaningful factors such as pose, color or identity are hidden.

- This motivates **latent variables**. Instead of modeling a complex p(x)p(x) directly

- we introduce a latent variable zz, model p(z)p(z), and generate data through p(x|z).

- A latent-variable model therefore represents complex data using simpler hidden structure.

- But latent variables create another problem. Computing
![[Pasted image 20260919161215.png]]
and especially the posterior p(z∣x)p(z|x) can be intractable.

- **Variational inference** solves this by introducing an approximate posterior qϕ(z∣x). Using Jensen's inequality, we derive the **ELBO**, with the key relationship:
![[Pasted image 20260919161406.png]]
Since KL divergence is non-negative, ELBO is a lower bound on the log-likelihood. This becomes the mathematical foundation of the **VAE**.

# 5 

A **VAE** consists of an encoder and decoder. The encoder maps x to the parameters μ,σ of a latent distribution; a latent sample z is then passed to the decoder to reconstruct/generate x. Training balances **reconstruction quality** with a **KL regularization term** that keeps the latent distribution close to the prior, commonly N(0,I).

Sampling creates a backpropagation problem, solved by the **reparameterization trick**:
![[Pasted image 20260919162352.png]]
This moves randomness into ϵ\epsilon while keeping the transformation differentiable with respect to μ,σ
# 6 

The second major approach is **GANs**. Instead of explicitly calculating likelihood, GANs compare real and generated samples through a **two-sample test**.

A generator G creates fake samples from noise z
A Discriminator D distinguishes real data from generated data

They are trained adversarially through a minimax objective.The mathematical analysis connects the original GAN objective to **Jensen-Shannon divergence** and more generally to f-divergences.

GANs are then extended into variants such as **cGAN, BiGAN, CoGAN and CycleGAN**
- cGAN adds conditions to control generation
- BiGAN combines generation with latent representation learning
- CoGAN handles related domains
- CycleGAN performs unpaired image-to-image translation using mappings in both directions and **cycle consistency**
- These variants show how the basic GAN framework can be adapted to different problems.

# 7 

The course also introduces other major architectures. **Autoregressive models** factorize the joint distribution into conditional probabilities and use likelihood-based learning

- **Diffusion models** learn generation through a gradual noising and denoising process.
- GANs, VAEs, diffusion models and autoregressive models represent different solutions to the same broad problem :
	- **learning and generating from complex data distributions**.

# 8 

how do we evaluate a generative model?

The answer depends on the goal.
- For density estimation : use likelihood or techniques such as **KDE** and importance sampling
- For Sample quality, we use **Inception Score, FID and KID/MMD**.
	- FID compares feature distributions of generated and test data using their means and covariances
	- KID applies MMD in feature space
# 9 

For latent representations 
evaluation focuses on 
- clustering
- compression
- disentanglement
A good representation should organize semantically related data, preserve information efficiently, and ideally separate meaningful independent factors

# 10 

So the complete subject can be remembered as:

Probability $\to$ MLE $\to$ Information Theory $\to$ Generative Modeling $\to$ Latent Variables $\to$ Variational Inference $\to$ ELBO $\to$ VAE.

and the parallel GAN path:
Generative Modeling $\to$ Two-Sample Testing $\to$ GAN $\to$ Divergences $\to$ GAN Variants

Both ultimately lead to:
Generative Models $\to$ Evaluation $\to$ Density + Sample Quality + Latent Representations


