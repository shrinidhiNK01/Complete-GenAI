# Generative AI: The Complete Masterclass (Beginner to Advanced)

Welcome to the ultimate guide to Generative AI (GenAI). This document explains every major generative model architecture, decoding strategy, and evaluation metric in simple English, followed by step-by-step implementation details.

---

## Table of Contents
1. **Introduction to Generative AI**
2. **Generative Adversarial Networks (GANs)**
3. **Variational Autoencoders (VAEs)**
4. **Diffusion Models & Stable Diffusion**
5. **LLM Text Decoding Strategies**
6. **Evaluating Generative Models (Text & Images)**

---

## 1. Introduction to Generative AI

### Discriminative vs. Generative Models
*   **Discriminative Models:** Learn the boundary between classes. They output a label $y$ given features $X$. E.g., *"Is this image a cat or a dog?"*
    $$\text{Math:} \quad P(y \mid X)$$
*   **Generative Models:** Learn the actual distribution of the data. They understand what a cat looks like as a whole, allowing them to generate *new* examples of cats.
    $$\text{Math:} \quad P(X) \quad \text{or} \quad P(X \mid y)$$

```text
Discriminative Model:
Image of Cat ===> [ Model ] ===> Label: "Cat" (Probability=0.98)

Generative Model:
Random Noise ===> [ Model ] ===> Brand New Image of a Cat
```

---

## 2. Generative Adversarial Networks (GANs)

Introduced by Ian Goodfellow in 2014, GANs consist of two neural networks playing a **minimax game** against each other.

### The Art Forger Analogy
*   **The Generator (The Forger):** Tries to create fake paintings (fake data) that look exactly like real ones.
*   **The Discriminator (The Art Detective):** Looks at paintings and decides if they are real (from the dataset) or fake (made by the Generator).

During training, the Forger gets better at making fakes, and the Detective gets better at spotting them. Eventually, the fakes are so good that the Detective is just guessing (50% accuracy).

```text
Real Images -----\
                  +---> [ Discriminator ] ---> Real or Fake?
Generator -> Fake/
(Noise)
```

### The Minimax Loss Function
The objective function is:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

1.  **$\max_{D}$:** The Discriminator wants $D(x)$ to be close to $1$ (spotting real data) and $D(G(z))$ to be close to $0$ (spotting fakes, meaning $1 - D(G(z)) \to 1$).
2.  **$\min_{G}$:** The Generator wants $D(G(z))$ to be close to $1$ (tricking the discriminator into thinking the fake is real, meaning $1 - D(G(z)) \to 0$).

### Key GAN Challenges
*   **Mode Collapse:** The Generator discovers a single output that always fools the Discriminator (e.g., generating only one specific face instead of a variety of faces) and keeps outputting *only* that sample.
*   **Training Instability:** If the Discriminator gets too smart too quickly, the Generator's gradients shrink to zero, and it stops learning.

---

## 3. Variational Autoencoders (VAEs)

An Autoencoder takes an input (e.g., an image), compresses it into a small bottleneck vector (latent space), and then reconstructs it. A **Variational Autoencoder (VAE)** forces the latent space to be continuous and smooth, which allows us to sample random points from it to generate *new* data.

```text
Input X ===> [ Encoder ] ===> Mean (µ) & StdDev (σ) ===> Sample z ===> [ Decoder ] ===> Reconstructed X'
```

### The Reparameterization Trick
To train a VAE, we need to backpropagate through the latent vector $z$. But $z$ is sampled randomly:

$$z \sim \mathcal{N}(\mu, \sigma^2)$$

Random sampling is non-differentiable (you cannot compute derivatives of a random roll). VAEs solve this by separating the random part:

$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

Here, $\epsilon$ is a static random node. Since $\mu$ and $\sigma$ are normal network weights, gradients can flow through them smoothly during backpropagation.

### The Loss Function
The VAE loss consists of two parts:

$$\text{Loss} = \text{Reconstruction Loss} + \text{KL Divergence}$$

1.  **Reconstruction Loss:** Measures how similar the output $X'$ is to the input $X$ (e.g. Mean Squared Error).
2.  **KL Divergence:** Measures how much the learned distribution ($\mu, \sigma$) deviates from a standard normal distribution $\mathcal{N}(0, I)$. This keeps the latent space centered and clustered, preventing gaps so we can generate smooth variations.

*See complete implementation in: `implementations/simple_vae.py`*

---

## 4. Diffusion Models & Stable Diffusion

Diffusion Models are the foundation of modern image generators (DALL-E 3, Midjourney, Stable Diffusion).

### How it works (The Analogy)
Imagine dropping a drop of blue ink into a glass of water.
*   **Forward Process:** Over time, the ink spreads out randomly until the water is completely blue noise. This is easy to do step-by-step.
*   **Reverse Process (Generative):** If we could play the video in reverse and collect all the scattered ink molecules back into a single drop, we would generate the drop from noise. The model learns this reverse process.

```text
Forward (Add Noise) ---> ---> ---> ---> ---> Noise (Gaussian)
Input Image <--- <--- <--- <--- <--- <--- Reverse (Predict & Subtract Noise)
```

### The Mathematics Simplified
1.  **Forward Process ($q$):** Gradually add Gaussian noise to an image $x_0$ over $T$ steps, ending with pure noise $x_T$.
2.  **Reverse Process ($p_\theta$):** A neural network (usually a **U-Net**) is trained at each step $t$ to look at the noisy image $x_t$ and predict the exact noise that was added to it.
3.  **Generation:** Start with random Gaussian noise $x_T$, let the U-Net predict the noise, subtract it, and repeat this step-by-step until a clean image $x_0$ emerges.

---

### Stable Diffusion (Latent Diffusion)
Running diffusion on high-resolution pixel images (e.g., $1024\times1024$) is extremely slow and expensive. Stable Diffusion solves this by performing the diffusion process in a compressed **Latent Space**.

It has three components:
1.  **Autoencoder (VAE):** Compresses the image into a smaller latent representation (e.g., $512\times512\times3$ pixels becomes $64\times64\times4$ latents).
2.  **Denoising U-Net:** Performs the forward/reverse noise addition and subtraction inside this smaller latent space.
3.  **Text Conditioning (CLIP):** A text encoder converts your prompt (e.g., *"A dog wearing a hat"*) into embeddings. These embeddings are injected into the U-Net via **Cross-Attention** layers to guide the image generation.

---

## 5. LLM Text Decoding Strategies

When an LLM generates text, it outputs a probability distribution over the entire vocabulary for the next word. Decoding strategies decide how to pick the word.

```text
Vocabulary probabilities:
"cat" (0.5), "dog" (0.3), "car" (0.15), "the" (0.05)
```

*   **Greedy Decoding:** Always pick the word with the absolute highest probability.
    *   *Pros:* Fast, deterministic.
    *   *Cons:* Repeats phrases easily, gets stuck in loops, lacks creativity.
*   **Beam Search:** Keeps track of the top $B$ (beam width) most likely sequences at each step instead of just one.
    *   *Pros:* High quality, used in translation.
    *   *Cons:* Computationally expensive, can sound dry/monotonous.
*   **Temperature Scaling:** Controls the randomness by dividing the raw scores (logits) before applying softmax.
    $$\text{Softmax}(z_i / T)$$
    *   **$T \to 0$:** Softmax becomes sharp, making the model pick only the top choice (greedy).
    *   **$T \to 1$:** Normal distribution.
    *   **$T > 1$:** Flattens the probabilities, making rare words likely to be picked (highly creative, but can output gibberish).
*   **Top-K Sampling:** Sort the vocabulary and only sample from the top $K$ most likely words.
*   **Top-P (Nucleus) Sampling:** Dynamically select the smallest set of words whose cumulative probability exceeds the threshold $P$ (e.g., $P=0.90$). If the model is confident, it samples from a few words; if it is uncertain, it samples from a wider range.

---

## 6. Evaluating Generative Models

### Text Generation Metrics
*   **BLEU (Bilingual Evaluation Understudy):** Compares generated text against human references by counting matching n-grams (sequences of words). Ranges from 0 to 1 (usually multiplied by 100). Primarily used in translation.
    *   *Limitation:* Focuses on exact word matches; ignores synonyms.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures the overlap of n-grams but focuses on **Recall** (how much of the reference text was captured by the generator). Standard for summarization.

---

### Image Generation Metrics
*   **FID (Fréchet Inception Distance):** Measures how realistic generated images are.
    1.  Pass a set of real images and generated images through a pre-trained image classifier (Inception-V3).
    2.  Extract the feature vectors from an intermediate layer.
    3.  Calculate the statistical difference (Fréchet distance) between the distributions of real and generated features.
    *   **Lower FID is better:** A score of $0$ means the generated image distribution is identical to the real image distribution.
