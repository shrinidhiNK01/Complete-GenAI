# Generative AI Images And Videos: Complete Simple Notes

This document explains how AI image and video generation works, especially diffusion models.

It connects with LLMs because modern generative AI systems often combine:

- Text encoders.
- Diffusion models.
- Transformers.
- Vision models.
- Multimodal models.

---

## 1. What Is Generative AI?

Generative AI creates new data.

Examples:

- Text.
- Images.
- Video.
- Audio.
- Code.
- 3D objects.

For images:

```text
Prompt: a cat wearing sunglasses
Output: new image matching the prompt
```

For video:

```text
Prompt: a car driving through rain at night
Output: sequence of frames showing that scene
```

---

## 2. Main Types Of Generative Models

### GANs

GAN means Generative Adversarial Network.

It has two networks:

Generator:

- Creates fake data.

Discriminator:

- Tries to detect fake data.

Training is a competition.

Problems:

- Hard to train.
- Mode collapse.
- Unstable.

### VAEs

VAE means Variational Autoencoder.

It learns a compressed latent space and can generate new samples.

Pros:

- Structured latent space.
- Stable training.

Cons:

- Outputs can be blurry.

### Diffusion Models

Diffusion models learn to remove noise step by step.

They are very important for modern image generation.

Pros:

- High image quality.
- Stable training.
- Good prompt control.

Cons:

- Generation can be slower.
- Video generation is expensive.

---

## 3. Diffusion Model Intuition

Training has two directions:

Forward process:

- Add noise to a real image step by step.

Reverse process:

- Learn to remove noise step by step.

Generation:

- Start with random noise.
- Repeatedly denoise.
- End with a clean image.

Simple analogy:

Imagine a clear photo slowly becoming static noise. A diffusion model learns how to reverse that static back into a meaningful image.

---

## 4. Forward Diffusion

Forward diffusion corrupts a real image.

At each step:

```text
slightly more noise is added
```

After many steps:

```text
image becomes almost pure noise
```

The model does not need to learn this part. We define the noise schedule.

---

## 5. Reverse Diffusion

Reverse diffusion is learned.

The model receives:

- Noisy image/latent.
- Timestep.
- Optional condition like text prompt.

It predicts:

- The noise to remove.

Then we subtract predicted noise step by step.

---

## 6. Noise Schedule

Noise schedule decides how much noise is added at each timestep.

Good schedule matters because:

- Too much noise too fast makes learning hard.
- Too little noise makes generation inefficient.

Common schedules:

- Linear.
- Cosine.

---

## 7. What Does The Neural Network Predict?

In diffusion, the neural network may predict:

- Noise.
- Clean image.
- Velocity-like target.

A common training objective is:

```text
predicted_noise should match actual_noise
```

Loss:

```text
MSE(predicted_noise, actual_noise)
```

---

## 8. U-Net

Many diffusion image models use a U-Net architecture.

U-Net has:

- Downsampling path.
- Bottleneck.
- Upsampling path.
- Skip connections.

Why useful:

- Downsampling captures global structure.
- Upsampling reconstructs details.
- Skip connections preserve local information.

---

## 9. Text-To-Image Generation

Text-to-image needs the model to understand prompt meaning.

High-level flow:

```text
text prompt -> tokenizer -> text encoder -> text embeddings
random noise -> denoising model guided by text -> image
```

The text encoder converts prompt into vectors.

The denoising model uses those vectors to guide image generation.

---

## 10. CLIP And Text Encoders

CLIP-style models learn a shared space for text and images.

They help connect:

- Prompt meaning.
- Visual concepts.

Text encoder output tells the diffusion model what to generate.

---

## 11. Cross-Attention In Image Generation

Text-guided diffusion often uses cross-attention.

Image latent features act like queries.

Text embeddings act like keys and values.

This lets image regions attend to prompt words.

Example:

Prompt:

```text
a red car beside a blue house
```

Cross-attention helps connect:

- `red` with car.
- `blue` with house.

---

## 12. Latent Diffusion

Pixel-space diffusion is expensive.

Latent diffusion compresses image into a smaller latent representation.

Flow:

```text
image -> VAE encoder -> latent
latent diffusion model denoises
latent -> VAE decoder -> image
```

Benefits:

- Faster.
- Less memory.
- High quality.

This idea powers many practical text-to-image systems.

---

## 13. Classifier-Free Guidance

Classifier-free guidance controls prompt strength.

During generation, model compares:

- Conditional prediction with prompt.
- Unconditional prediction without prompt.

Guidance scale controls how strongly prompt affects result.

High guidance:

- More prompt adherence.
- May look less natural or overcooked.

Low guidance:

- More natural/random.
- May ignore prompt details.

---

## 14. Image-To-Image

Image-to-image starts with an existing image.

Process:

1. Encode image.
2. Add some noise.
3. Denoise guided by prompt.

Noise strength controls how much image changes.

Low strength:

- Keeps original structure.

High strength:

- Changes image strongly.

---

## 15. Inpainting

Inpainting edits only part of an image.

Input:

- Original image.
- Mask showing area to change.
- Text prompt.

The model fills masked region while preserving rest.

Uses:

- Remove object.
- Add object.
- Fix face/hands.
- Change background part.

---

## 16. Outpainting

Outpainting extends image beyond original borders.

Example:

- Make a portrait into a wide landscape scene.

Model generates new content around the existing image.

---

## 17. ControlNet And Conditioning

ControlNet adds extra control signals.

Examples:

- Edge map.
- Pose skeleton.
- Depth map.
- Segmentation mask.
- Scribble.

Why useful:

- Text alone may not control layout.
- Control signals guide structure.

---

## 18. Video Generation

Video is a sequence of images.

But generating each frame independently fails because:

- Character changes frame to frame.
- Background flickers.
- Motion is inconsistent.

Video models must learn:

- Spatial quality.
- Temporal consistency.
- Motion.
- Object permanence.

---

## 19. Video Diffusion

Video diffusion extends diffusion to time.

Instead of denoising:

```text
height x width x channels
```

it denoises:

```text
frames x height x width x channels
```

This is much more expensive.

---

## 20. Temporal Attention

Video models may use attention across:

- Space: pixels/patches within a frame.
- Time: same object across frames.

Temporal attention helps maintain consistency.

Example:

- A red ball should stay red across frames.
- A person should keep same clothes.

---

## 21. Text-To-Video

Text-to-video flow:

```text
prompt -> text encoder
random video noise -> video denoising model
denoised latent frames -> decoder -> video
```

Challenges:

- Long-range motion.
- Consistent identity.
- Physics.
- Camera movement.
- High resolution.
- Long duration.

---

## 22. Image-To-Video

Image-to-video starts with an image and animates it.

Input:

- Starting image.
- Optional prompt.
- Optional motion instruction.

The model generates future frames.

Challenge:

- Keep identity while creating realistic motion.

---

## 23. Text, Image, And Video Together

Modern systems can use multiple inputs:

- Text prompt.
- Reference image.
- Pose.
- Depth.
- Audio.
- Previous frames.

These are conditioning signals.

The model uses them to control generation.

---

## 24. Evaluation Of Image And Video Models

Image evaluation:

- Human preference.
- Prompt alignment.
- Visual quality.
- Diversity.
- Artifact rate.

Metrics:

- FID.
- CLIP score.
- Aesthetic score.

Video evaluation:

- Frame quality.
- Motion quality.
- Temporal consistency.
- Prompt alignment.
- Human preference.

Metrics are imperfect. Human evaluation is still very important.

---

## 25. Common Failure Modes

Images:

- Weird hands.
- Text rendering mistakes.
- Object count mistakes.
- Prompt details ignored.
- Mixed attributes.
- Unnatural lighting.

Videos:

- Flickering.
- Identity drift.
- Object disappearing.
- Unrealistic physics.
- Warped motion.
- Scene inconsistency.

---

## 26. Safety And Ethics

Risks:

- Deepfakes.
- Copyright issues.
- Misinformation.
- Non-consensual imagery.
- Bias in generated outputs.

Mitigations:

- Watermarking.
- Content filters.
- Usage policies.
- Provenance metadata.
- Human review.
- Dataset filtering.

---

## 27. Simple Interview Answers

### How do diffusion models work?

They learn to remove noise. During training, noise is added to real images, and the model learns to predict that noise. During generation, the model starts from random noise and denoises step by step until an image appears.

### Why latent diffusion?

Latent diffusion works in compressed image space instead of pixel space, making generation faster and cheaper while preserving quality.

### What is classifier-free guidance?

It controls how strongly the generation follows the prompt by combining conditional and unconditional denoising predictions.

### Why is video generation harder than image generation?

Video must be consistent across time. The model must generate good frames and realistic motion without flickering or changing object identity.

### What is ControlNet?

ControlNet adds extra structure signals like pose, depth, edges, or masks to control image generation more precisely.

---

## 28. Final Summary

Image diffusion:

```text
real image -> add noise -> train model to remove noise
random noise -> remove noise step by step -> generated image
```

Text-to-image:

```text
prompt -> text embedding -> guided denoising -> image
```

Video generation:

```text
prompt/image -> guided denoising across space and time -> video
```

The key idea:

Generative image/video models learn the structure of visual data by learning how to reverse noise.

