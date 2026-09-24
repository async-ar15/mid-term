Here is the Generative Artificial Intelligence Mid-Semester Examination question paper extracted from the document:

---

### **Rajiv Gandhi Institute of Petroleum Technology**

**(An Institute of National Importance)**

**Mid-Semester Examination, Odd Semester, 2025–2026**

**Programme:** B. Tech. 4th YEAR $(CSE + IT + CSD + IDD)$

**Course Name:** Generative Artificial Intelligence

**Course Code:** CS441

**Time:** 2 Hours | **Max. Marks:** 30

---

#### **Instructions:**

1. Do not write anything other than your roll number on the question paper.


2. Assume suitable data wherever essential and mention it clearly.


3. Writing appropriate units, nomenclature, and drawing neat sketches/schematics wherever required is an integral part of the answer.



---

#### **Questions**

* **Q1:** Explain the core idea of adversarial training in Generative Adversarial Networks with its architecture and training objectives. Justify why GAN training is expressed as a min-max optimization problem. Suppose in a GAN training, a mini-batch for an iteration is given as:


* Real samples: $D(x) = [0.9, 0.8, 0.7, 0.95]$

* Generated samples: $D(G(z)) = [0.2, 0.3, 0.1, 0.4]$



Considering the above, solve the following questions: **[10 Marks]**

1. Compute the discriminator loss $L_D$ for this batch.


2. Compute the generator loss $L_G$ for the output sample.


3. Interpret what these values imply about the discriminator and generator's performance at this training step.




**OR**
* **Q1:** Explain the major challenges of non-convergence and mode-collapse faced during the training of Generative Adversarial Networks, illustrating with numerical examples. Discuss in detail any two GAN variants proposed to overcome the mentioned challenges. **[10 Marks]**



---

* **Q2:** Why are VAEs considered generative models, while vanilla autoencoders are not? What is the reparameterization trick in VAE, how and why is it implemented? Given $\mu = [0.5, -1.0]$, $\sigma = [0.2, 0.5]$ and a random sample $\epsilon = [0.1, -0.3]$ from $\mathcal{N}(0, 1)$, compute the latent vector $z$ using the reparameterization trick. Assume the parameters whenever required. **[10 Marks]**


**OR**
* **Q2:** Why does VAE use $KL(q(z\vert{}x) \parallel p(z))$ instead of $KL(p(z) \parallel q(z\vert{}x))$? The figure illustrates the idea of variational inference (VI). The ellipse represents a family of approximate distributions $q(z; v)$ parameterized by $v$. The black dot outside the ellipse corresponds to the true posterior $p(z\vert{}x)$. Optimization starts from $v^{\text{init}}$ and converges to $v^*$, the optimal parameters within the family. Illustrate this process of the VI and answer the questions below: **[10 Marks]**


* **A.** Explain why the true posterior $p(z\vert{}x)$ generally lies outside the variational family $q(z; v)$.


* **B.** Show how minimizing $KL(q(z; v) \parallel p(z\vert{}x))$ is equivalent to maximizing the Evidence Lower Bound (ELBO).


* **C.** What does the path from $v^{\text{init}}$ to $v^*$ represent? Interpret the role of $v^{\text{init}}$ and $v^*$ in the optimization procedure.





---

* **Q3:** Explain the complete training and sampling pipeline of a diffusion model. A Denoising Diffusion Probabilistic Model (DDPM) is defined by a forward noising process:


$$
q(x_t \vert{} x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})
$$



* **A.** What is $\beta_t$ and its significance in the diffusion model?


* **B.** Formulate the equation of $x_t$ as a Markov process. Why does $x_T$ converge to Gaussian noise as $t \to T$?


* **C.** Write the mathematical representation of the generative process $p_\theta(x_{t-1}\vert{}x_t)$ in the diffusion model and how we can relate it with the variational inference of VAE?


* **D.** Represent both the Forward and Reverse Diffusion Process in Stochastic Differential Equations (SDE). **[10 Marks]**




**OR**
* **Q3:** Diffusion models, while highly effective for generative tasks, encounter two key challenges: slow sampling and scalability to high-resolution (conditional) generation. State and explain different strategies adopted in diffusion models with respect to the following: **[10 Marks]**


* **A.** Discuss the strategies employed to accelerate the sampling process in diffusion models. Explain their underlying working principles and architectural approaches wherever applicable.


* **B.** Explain how diffusion models can be adapted for high-resolution conditional generation. Illustrate the architectural techniques and guidance mechanisms used.