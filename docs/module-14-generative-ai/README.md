# Module 14: Generative AI Beyond LLMs — Diffusion, Vision & Audio

> **Prerequisites:** Module 1 (Foundations), Module 13 (CNNs, GANs, VAEs)  
> **Estimated Time:** 8-10 hours  
> **Relevance:** Modern AI systems are multimodal. Understanding image, video, and audio generation is essential for building complete AI applications

---

## 14.1 The Generative AI Landscape

```
┌──────────────────────────────────────────────────────────────────┐
│                    Generative AI Model Types                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  TEXT:       LLMs (GPT, Claude, LLaMA)      ← Modules 1-10      │
│                                                                   │
│  IMAGES:    Diffusion Models (DALL-E 3, Midjourney, SD, Flux)    │
│             GANs (StyleGAN — legacy) ← THIS MODULE               │
│                                                                   │
│  VIDEO:     Sora (OpenAI), Veo 2 (Google), Runway Gen-3          │
│             Based on video diffusion transformers                  │
│                                                                   │
│  AUDIO:     Music: Suno, Udio, MusicGen                          │
│             Speech: Whisper (STT), TTS models                     │
│             Sound: AudioLDM, Stable Audio                         │
│                                                                   │
│  3D:        Point-E, Shap-E, Gaussian Splatting                  │
│                                                                   │
│  CODE:      Codex, StarCoder, DeepSeek-Coder (also LLMs)        │
│                                                                   │
│  MULTIMODAL: GPT-4o, Gemini, Claude (unified input/output)       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 14.2 Diffusion Models — The Foundation of Image Generation

### Core Intuition

```
Forward Process (add noise gradually):

  Clean image → Slightly noisy → More noisy → ... → Pure noise
     x₀           x₁              x₂              x_T ~ N(0,I)

  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │ 🐱   │ → │ 🐱+ε │ → │ 🐱+εε│ → │ ░░░░ │ → │ ████ │
  │ cat  │   │      │   │      │   │      │   │noise │
  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
   t=0        t=1        t=2       t=T-1       t=T

Reverse Process (LEARN to denoise):

  Pure noise → Less noisy → ... → Slightly noisy → Clean image!
     x_T          x_{T-1}           x₁               x₀

  The model learns: given noisy image at step t,
  predict the NOISE that was added (ε-prediction)

  Then subtract that predicted noise to get a cleaner image.
```

### Mathematical Framework (DDPM)

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

```
Forward Process (fixed, not learned):
  q(xₜ | xₜ₋₁) = N(xₜ ; √(1-βₜ) xₜ₋₁, βₜI)

  Noise schedule: β₁ < β₂ < ... < β_T  (small to large noise)

  Shortcut (jump to any timestep):
  q(xₜ | x₀) = N(xₜ ; √ᾱₜ x₀, (1-ᾱₜ)I)
  where ᾱₜ = Π(1-βᵢ) for i=1..t

  → xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  ε ~ N(0,I)

Reverse Process (learned neural network):
  pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁ ; μθ(xₜ, t), σₜ²I)

  The network εθ predicts the noise:
  μθ(xₜ, t) = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ)) εθ(xₜ, t))

Training Loss (simplified):
  L = E[‖ε - εθ(xₜ, t)‖²]

  "Predict the noise that was added" — that's the entire training!
```

### The U-Net Architecture for Diffusion

```
U-Net: The workhorse architecture for noise prediction

Input: noisy image xₜ + timestep t
Output: predicted noise εθ

     ┌──────────────────────────────────────────┐
     │            Skip Connections               │
     │   ┌─────────────────────────────────┐    │
     │   │                                 │    │
    64 ──┤ Down ──128 ── Down ──256 ── Mid ──256 ── Up ──128 ── Up ──64
     │   │  │               │                        │          │
     │   │  └───────────────┼────────────────────────┘          │
     │   │                  └───────────────────────────────────┘
     │   └─────────────────────────────────────────────────────────── Out
     │
     │  + Time embedding (sinusoidal, added/concat at each block)
     │  + Text embedding (cross-attention from text encoder)
     └──────────────────────────────────────────┘

Components:
  ResNet blocks:     Convolution + normalization + activation
  Self-attention:    Attend to other spatial positions (global context)
  Cross-attention:   Attend to text embeddings (conditioning)
  Time embedding:    Sinusoidal position encoding for timestep t
  Skip connections:  U-Net's hallmark — preserve high-res details
```

### From DDPM to Modern Diffusion

```
Evolution:

DDPM (2020):     1000 denoising steps → slow!
                 ~30 seconds per image

DDIM (2020):     Deterministic sampling, skip steps
                 50-100 steps sufficient → faster

Classifier       Guide diffusion with class labels
Guidance (2021): Scale: w × ε_conditional + (1-w) × ε_unconditional

Classifier-Free  Train with/without conditioning (random dropout)
Guidance (2022): No separate classifier needed
                 CFG scale controls adherence to prompt (7-15 typical)

                 ε̃ = ε_uncond + s × (ε_cond - ε_uncond)
                 s > 1: stronger prompt adherence (less diversity)
                 s = 1: standard conditional generation

Latent Diffusion  Diffuse in LATENT space, not pixel space!
(Stable Diff):    Image → VAE encoder → latent → diffuse → VAE decode
                  64×64 latent instead of 512×512 pixels = 64× cheaper

                  ┌─────────┐   ┌───────────┐   ┌─────────┐
                  │  Text   │   │  Latent   │   │   VAE   │
                  │ Encoder │──▶│ Diffusion │──▶│ Decoder │──▶ Image
                  │ (CLIP)  │   │  (U-Net)  │   │         │
                  └─────────┘   └───────────┘   └─────────┘
```

---

## 14.3 Text-to-Image Systems

### DALL-E Evolution

```
DALL-E 1 (2021):  Discrete VAE + autoregressive transformer
                  Tokenize image into 32×32 grid of discrete tokens
                  Generate tokens autoregressively (like text!)

DALL-E 2 (2022):  CLIP text encoder → prior (diffusion) → decoder (diffusion)
                  Two-stage: text → CLIP image embedding → pixels
                  Uniter: CLIP connects text and image spaces

DALL-E 3 (2023):  Much better prompt following
                  Key innovation: Improved training data with
                  synthetic captions (re-caption all training images
                  with a captioning model for better text alignment)
                  Integrated with ChatGPT for prompt rewriting
```

### Stable Diffusion Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Stable Diffusion Pipeline                            │
│                                                                   │
│  "A photo of a cat wearing a top hat, oil painting style"        │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────┐                                                │
│  │ Text Encoder │  CLIP ViT-L/14 or OpenCLIP                    │
│  │ (frozen)     │  Tokenize → 77 tokens → 768-dim embeddings    │
│  └──────┬───────┘                                                │
│         │ text embeddings                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────┐                            │
│  │      U-Net (noise predictor)    │                             │
│  │                                  │                             │
│  │  Input: noisy latent (4×64×64)  │                             │
│  │  + timestep embedding           │                             │
│  │  + text cross-attention         │     ← text controls image   │
│  │                                  │                             │
│  │  Iteratively denoise:           │                             │
│  │  z_T → z_{T-1} → ... → z_0     │                             │
│  └──────┬───────────────────────────┘                            │
│         │ cleaned latent z₀                                      │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │ VAE Decoder  │  Latent (4×64×64) → Image (3×512×512)         │
│  │ (frozen)     │                                                │
│  └──────────────┘                                                │
│                                                                   │
│  Total: ~1B parameters (U-Net) + encoders/decoders               │
└──────────────────────────────────────────────────────────────────┘
```

### Diffusion Transformers (DiT) — The Future

```
Evolution: U-Net → Transformer

Why DiT?
  U-Net was borrowed from segmentation → not designed for generation
  Transformers scale better with data and compute (scaling laws!)

DiT Architecture:
  Replace U-Net with a Vision Transformer:

  Noisy latent patches    +  Time embedding  +  Text conditioning
         │                        │                   │
         ▼                        ▼                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │            Transformer Blocks (DiT Block)                │
  │                                                          │
  │  [AdaLN] → [Self-Attention] → [Cross-Attention] → [FFN] │
  │                                                          │
  │  AdaLN: Adaptive Layer Norm (condition on time + text)   │
  │  Works just like a decoder-only transformer!             │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Predicted noise

Used in: DALL-E 3, Stable Diffusion 3, Flux, Sora
```

### CLIP — Connecting Language and Vision

```
CLIP Training (Contrastive Language-Image Pre-training):

  Batch of (image, text) pairs:

  Image Encoder    Text Encoder
  (ViT or ResNet)  (Transformer)
       │                │
       ▼                ▼
  Image embeddings  Text embeddings
      [I₁]              [T₁]     ← should be close (matching pair)
      [I₂]              [T₂]     ← should be close
      [I₃]              [T₃]     ← should be close

  Similarity matrix:
           T₁    T₂    T₃
    I₁  [ HIGH  low   low  ]    maximize diagonal
    I₂  [ low   HIGH  low  ]    minimize off-diagonal
    I₃  [ low   low   HIGH ]

  Loss: InfoNCE (contrastive) — symmetric cross-entropy

  Result: Shared embedding space for text AND images

  Applications:
    - Image-text matching (search by description)
    - Zero-shot image classification (no training needed!)
    - Vision encoder for multimodal LLMs (LLaVA, GPT-4V)
    - Guiding diffusion models (text conditioning)
```

---

## 14.4 Image Editing & Controlled Generation

```
┌──────────────────────────────────────────────────────────────────┐
│              Controlled Generation Techniques                     │
├──────────────┬──────────────────────────────────────────────────┤
│ Img2Img      │ Start from partially noised source image         │
│              │ instead of pure noise → preserves structure      │
├──────────────┼──────────────────────────────────────────────────┤
│ Inpainting   │ Mask part of image → only generate in masked     │
│              │ region → edit specific regions                   │
├──────────────┼──────────────────────────────────────────────────┤
│ ControlNet   │ Additional conditioning with spatial controls:    │
│              │ edge maps, depth maps, pose, segmentation        │
│              │ Preserves composition while changing style        │
├──────────────┼──────────────────────────────────────────────────┤
│ IP-Adapter   │ Image prompt adapter — use a reference image     │
│              │ as a style/content guide                          │
├──────────────┼──────────────────────────────────────────────────┤
│ LoRA for     │ Fine-tune diffusion model with ~100 images       │
│ Diffusion    │ of a specific style/character/concept            │
│              │ Same low-rank adaptation as for LLMs!            │
├──────────────┼──────────────────────────────────────────────────┤
│ Textual      │ Learn a new "word" embedding for a concept       │
│ Inversion    │ from few images — <<my_style>> in prompts        │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 14.5 Video Generation

```
Video = Image sequence with temporal consistency

Challenges:
  1. Temporal coherence (objects don't flicker/teleport)
  2. Compute cost (3D data: height × width × time × channels)
  3. Motion understanding (physics, gravity, dynamics)
  4. Long-range consistency (beginning matches end)

Architecture Approaches:

  ┌──────────────────────────────────────────────────────────┐
  │  Sora (OpenAI, 2024):                                    │
  │                                                           │
  │  Video → "spacetime patches" (3D tokens)                  │
  │  Diffusion Transformer in latent space                    │
  │  Can generate variable resolution/aspect ratio/duration   │
  │  Trained on massive video + image data                    │
  │                                                           │
  │  3D Patch Tokenization:                                   │
  │  ┌────┬────┐  ┌────┬────┐  ┌────┬────┐                  │
  │  │ P1 │ P2 │  │ P5 │ P6 │  │ P9 │P10 │                  │
  │  ├────┼────┤  ├────┼────┤  ├────┼────┤                  │
  │  │ P3 │ P4 │  │ P7 │ P8 │  │P11 │P12 │                  │
  │  └────┴────┘  └────┴────┘  └────┴────┘                  │
  │   Frame 1      Frame 2      Frame 3                      │
  │                                                           │
  │  Each patch spans space AND time                          │
  │  → Temporal attention built into transformer              │
  └──────────────────────────────────────────────────────────┘

Key Models (2024-2026):
  Sora:       Up to 1 minute, photorealistic
  Veo 2:      Google, 4K resolution, good physics
  Runway Gen-3: Commercial, fast iteration
  Kling:      Kuaishou, long video generation
  Wan:        Open source video generation
```

---

## 14.6 Audio & Speech AI

### Speech-to-Text (Whisper)

```
Whisper Architecture (OpenAI):

  Audio (mel spectrogram)
    │
    ▼
  ┌──────────────────┐        ┌──────────────────┐
  │  Audio Encoder   │───────▶│  Text Decoder    │
  │  (Transformer)   │ cross  │  (Transformer)   │
  │                  │ attn   │                  │
  │  Log-mel spec    │        │  Autoregressive   │
  │  → 2D conv       │        │  text generation  │
  │  → transformer   │        │                  │
  └──────────────────┘        └──────────────────┘
                                       │
                                       ▼
                              "Hello, how are you?"

Key features:
  - Encoder-decoder (like T5, not like GPT)
  - Trained on 680K hours of labeled audio
  - Multilingual (100+ languages)
  - Handles accents, noise, varied audio quality
  - Can translate speech directly (any language → English)

  Special tokens: <|startoftranscript|>, <|en|>, <|transcribe|>
  Multi-task: transcribe, translate, timestamp, language detect
```

### Text-to-Speech (TTS)

```
Modern TTS Pipeline:

  Text → [Text Encoder] → [Duration/Acoustic Model] → [Vocoder] → Audio

  Evolution:
    Concatenative (2000s):  Stitch pre-recorded clips
    Tacotron 2 (2017):      Seq2seq → mel spectrogram → WaveNet
    VITS (2021):            End-to-end, real-time, VAE + flow
    VALL-E (2023):          Neural codec language model
                            "GPT for speech" — 3 second voice clone!

  VALL-E Architecture:
    1. Audio → Neural audio codec → Discrete tokens
       (like tokenizing text, but for audio)
    2. Text + 3-second voice sample → predict audio tokens
       (autoregressive, like a language model!)
    3. Audio tokens → codec decoder → waveform

    This is fundamentally an LLM-style approach applied to audio.
```

### Music Generation

```
MusicGen (Meta, 2023):
  Text/melody → music
  Architecture: Transformer LM over audio tokens
  Encodec audio codec → discrete tokens → autoregressive generation

Suno / Udio (2024):
  Text → complete songs with vocals
  End-to-end, commercial quality
  Combine music generation + vocal synthesis

Key insight:
  The same autoregressive transformer paradigm works for
  text, code, images (DALL-E 1), audio, and music!
  → "Everything is a sequence of tokens"
```

---

## 14.7 3D Generation & Beyond

```
3D Generation Approaches:

  Text → 3D:
    Point-E (OpenAI):      Text → point cloud → mesh
    Shap-E (OpenAI):       Text → implicit 3D representation
    DreamFusion (2022):    Optimize NeRF using 2D diffusion prior

  Image → 3D:
    Zero-1-to-3:           Single image → 3D object
    Gaussian Splatting:     Fast 3D from multi-view images

  NeRF (Neural Radiance Fields):
    Input: multiple photos of a scene
    Output: 3D representation (render from any angle)
    Learn: f(x, y, z, θ, φ) → (color, density)

  3D Gaussian Splatting (2023):
    Faster than NeRF (real-time rendering!)
    Represent scene as millions of 3D gaussians
    Each gaussian: position, covariance, color, opacity
    Differentiable rasterization for training
```

---

## 14.8 Key Concepts Across Generative AI

### Latent Spaces

```
Every generative model operates in a latent space:

  VAE:        Image → encoder → z (continuous, regularized) → decoder → image
  GAN:        z (random) → generator → image
  Diffusion:  Image → VAE → latent → noise/denoise → VAE → image
  LLM:        Token → embedding → hidden states → output distribution

  Why latent spaces matter:
    - Compression: operate in lower dimensions (cheaper!)
    - Interpolation: smooth transitions between concepts
    - Disentanglement: separate factors of variation
    - Composition: combine concepts (style + content)

  Latent interpolation:

  z_cat ──── z_blend₁ ──── z_blend₂ ──── z_dog
   🐱          🐱🐕          🐕🐱          🐕

  Smooth transition in latent space → smooth transition in output
```

### The Tokenization Pattern

```
Unifying principle: EVERYTHING can be tokenized

  Text:    Words/subwords → token IDs → embeddings
  Images:  Patches (ViT) or VAE latents → continuous/discrete tokens
  Audio:   Mel spectrogram → codec tokens (EnCodec, SoundStream)
  Video:   Spacetime patches → tokens
  3D:      Point clouds or voxels → tokens
  Music:   Audio codec tokens (same as speech)
  Code:    Same as text (subword tokenization)
  Actions: Discretized continuous actions → tokens (RT-2)

  → Once tokenized, the SAME transformer architecture works!
  → This is why LLM techniques (attention, scaling, RLHF) transfer
```

### Evaluation Metrics for Generative Models

```
┌──────────────┬──────────────────────────────────────────────────┐
│ FID (Fréchet │ Compare statistics of generated vs real images    │
│ Inception    │ Lower = more realistic                           │
│ Distance)    │ Standard metric for image generation             │
├──────────────┼──────────────────────────────────────────────────┤
│ CLIP Score   │ Cosine similarity between generated image and    │
│              │ text prompt in CLIP space                        │
│              │ Higher = better text-image alignment             │
├──────────────┼──────────────────────────────────────────────────┤
│ Inception    │ Quality (KL divergence of class predictions)     │
│ Score (IS)   │ Higher = more realistic AND diverse              │
├──────────────┼──────────────────────────────────────────────────┤
│ LPIPS        │ Perceptual similarity using deep features        │
│              │ Lower = more perceptually similar                │
├──────────────┼──────────────────────────────────────────────────┤
│ Human Eval   │ Still the gold standard — Elo ratings from       │
│              │ pairwise human preferences (like Chatbot Arena)  │
├──────────────┼──────────────────────────────────────────────────┤
│ WER (audio)  │ Word Error Rate for speech recognition           │
│              │ Lower = better transcription                     │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 14.9 Practical Implementation

<details>
<summary><strong>Complete Code: Simplified Diffusion Model</strong></summary>

```python
import numpy as np

class SimpleDiffusion:
    """
    Minimal diffusion model to understand the core algorithm.
    Uses a simple MLP as the noise predictor (real models use U-Net/DiT).
    """

    def __init__(self, data_dim=2, T=100):
        self.T = T  # number of diffusion steps
        self.data_dim = data_dim

        # Noise schedule (linear)
        self.betas = np.linspace(0.0001, 0.02, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # Simple MLP noise predictor
        hidden = 128
        self.W1 = np.random.randn(data_dim + 1, hidden) * 0.1  # +1 for timestep
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * 0.1
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, data_dim) * 0.1
        self.b3 = np.zeros(data_dim)

    def predict_noise(self, x_t, t):
        """Neural network that predicts the noise added at timestep t."""
        # Normalize timestep to [0, 1]
        t_norm = np.array([[t / self.T]])
        inp = np.concatenate([x_t.reshape(1, -1), t_norm], axis=1)

        h = np.maximum(0, inp @ self.W1 + self.b1)  # ReLU
        h = np.maximum(0, h @ self.W2 + self.b2)     # ReLU
        return (h @ self.W3 + self.b3).flatten()

    def forward_diffusion(self, x_0, t):
        """Add noise to x_0 according to schedule at timestep t."""
        alpha_bar = self.alpha_bars[t]
        noise = np.random.randn(*x_0.shape)
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def train_step(self, x_0, lr=0.001):
        """One training step: predict the noise that was added."""
        # Random timestep
        t = np.random.randint(0, self.T)

        # Add noise
        x_t, true_noise = self.forward_diffusion(x_0, t)

        # Predict noise
        pred_noise = self.predict_noise(x_t, t)

        # Loss = MSE between true and predicted noise
        loss = np.mean((true_noise - pred_noise) ** 2)

        # Simplified gradient update (in practice, use autograd)
        # This is just to show the concept
        return loss

    def sample(self):
        """Generate a new sample by iterative denoising."""
        # Start from pure noise
        x = np.random.randn(self.data_dim)

        # Iteratively denoise
        for t in range(self.T - 1, -1, -1):
            predicted_noise = self.predict_noise(x, t)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            # Denoising step (DDPM equation)
            x = (1 / np.sqrt(alpha)) * (
                x - (self.betas[t] / np.sqrt(1 - alpha_bar)) * predicted_noise
            )

            # Add noise (except at last step)
            if t > 0:
                noise = np.random.randn(self.data_dim)
                x += np.sqrt(self.betas[t]) * noise

        return x

# ============================================================
# CLIP-STYLE CONTRASTIVE LEARNING
# ============================================================

class SimpleCLIP:
    """Simplified CLIP to understand contrastive learning."""

    def __init__(self, image_dim, text_dim, embed_dim):
        self.image_proj = np.random.randn(image_dim, embed_dim) * 0.01
        self.text_proj = np.random.randn(text_dim, embed_dim) * 0.01
        self.temperature = 0.07  # learned in real CLIP

    def encode_image(self, images):
        """Project images to shared embedding space."""
        embeddings = images @ self.image_proj
        # L2 normalize
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def encode_text(self, texts):
        """Project texts to shared embedding space."""
        embeddings = texts @ self.text_proj
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def contrastive_loss(self, images, texts):
        """InfoNCE loss — the CLIP training objective."""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)

        # Similarity matrix (batch × batch)
        logits = (image_embeds @ text_embeds.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = len(images)
        labels = np.arange(batch_size)

        # Symmetric cross-entropy loss
        # Image → Text direction
        loss_i2t = -np.mean(logits[range(batch_size), labels] -
                            np.log(np.sum(np.exp(logits), axis=1)))
        # Text → Image direction
        loss_t2i = -np.mean(logits.T[range(batch_size), labels] -
                            np.log(np.sum(np.exp(logits.T), axis=1)))

        return (loss_i2t + loss_t2i) / 2

    def zero_shot_classify(self, image, class_texts):
        """Zero-shot classification using CLIP."""
        image_embed = self.encode_image(image.reshape(1, -1))
        text_embeds = self.encode_text(class_texts)
        similarities = (image_embed @ text_embeds.T).flatten()
        return np.argmax(similarities)

# Example
diffusion = SimpleDiffusion(data_dim=2, T=50)
print(f"Generated sample: {diffusion.sample()}")
```

</details>

---

## 14.10 Interview Questions

### Conceptual Questions

**Q1: Explain how diffusion models generate images. What are the forward and reverse processes?**

Forward process: Gradually add Gaussian noise to an image over T steps according to a fixed schedule ($\beta_1 < ... < \beta_T$). At step T, the image is pure noise. This is deterministic and doesn't require training. Reverse process: A neural network (U-Net or DiT) learns to predict the noise added at each step. Starting from pure noise, iteratively predict and remove noise over T steps to generate a clean image. Training loss: MSE between actual noise and predicted noise: $L = E[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$. Key insight: by learning to denoise at every noise level, the model implicitly learns the data distribution.

**Q2: What is classifier-free guidance (CFG) and why is it important for text-to-image generation?**

CFG trains the diffusion model to work both WITH and WITHOUT text conditioning (randomly dropping the text prompt during training, e.g., 10% of the time). At inference, compute two predictions: $\epsilon_\text{cond}$ (with text) and $\epsilon_\text{uncond}$ (without text). The final prediction is: $\tilde{\epsilon} = \epsilon_\text{uncond} + s(\epsilon_\text{cond} - \epsilon_\text{uncond})$, where $s > 1$ amplifies the text signal. Higher s = stronger prompt adherence but less diversity. Typical values: s = 7-15. This replaced separate classifier guidance, simplifying the pipeline and improving quality.

**Q3: How does CLIP work, and why is it foundational to modern multimodal AI?**

CLIP trains an image encoder and text encoder jointly using contrastive learning on 400M image-text pairs. Goal: matching image-text pairs should have high cosine similarity; non-matching pairs should be low. Uses InfoNCE loss on the similarity matrix (symmetric cross-entropy on the batch×batch similarity matrix). Result: a shared embedding space where text and images are aligned. Foundational because: (1) zero-shot classification (no task-specific training), (2) text conditioning for diffusion models, (3) vision encoder for multimodal LLMs, (4) image-text retrieval, (5) evaluation metric (CLIP Score).

**Q4: Compare GANs and diffusion models. Why have diffusion models largely replaced GANs?**

GANs: faster sampling (single forward pass), but suffer from mode collapse (limited diversity), training instability (adversarial game), and no likelihood computation. Diffusion models: slower sampling (many denoising steps), but offer stable training (simple MSE loss), better diversity (no mode collapse), controllable generation (CFG, ControlNet), and log-likelihood estimation. Diffusion won because: (1) better FID scores at scale, (2) much easier to train, (3) natural support for text conditioning via cross-attention, (4) composability (inpainting, editing, ControlNet). Speed gap is closing with DDIM, consistency models, and distillation (1-4 step generation).

**Q5: Explain the "latent" in Latent Diffusion Models. Why is it important?**

Instead of running diffusion in pixel space (3×512×512 = 786K dimensions), first compress images to a latent space using a pre-trained VAE (4×64×64 = 16K dimensions, a 49×reduction). Diffusion then operates on these compact latents. Benefits: (1) massive compute savings (49× fewer dimensions), (2) VAE removes imperceptible high-frequency details that waste model capacity, (3) enables higher resolution generation, (4) latent space has more semantic structure (nearby latents = similar images). This is what made Stable Diffusion practical to run on consumer GPUs.

### Coding Questions

**Q6: Implement the core diffusion sampling loop (DDPM) given a trained noise predictor.**

```python
def ddpm_sample(noise_predictor, shape, T, betas):
    """Generate a sample using DDPM reverse process."""
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    # Start from pure Gaussian noise
    x = np.random.randn(*shape)

    for t in range(T - 1, -1, -1):
        # Predict noise
        eps_pred = noise_predictor(x, t)

        # Compute denoised estimate
        coeff1 = 1 / np.sqrt(alphas[t])
        coeff2 = betas[t] / np.sqrt(1 - alpha_bars[t])

        x = coeff1 * (x - coeff2 * eps_pred)

        # Add noise for all steps except the last
        if t > 0:
            sigma = np.sqrt(betas[t])
            x += sigma * np.random.randn(*shape)

    return x
```

### System Design Questions

**Q7: Design a text-to-image generation service that handles 100 requests per second with <5 second latency.**

```
┌──────────────────────────────────────────────────────────────────┐
│           Text-to-Image Service Architecture                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. API LAYER                                                     │
│     REST/WebSocket API, rate limiting per user                    │
│     Prompt safety filter (block NSFW/harmful prompts)             │
│     Queue: Redis priority queue (paid users first)                │
│                                                                   │
│  2. PROMPT PROCESSING                                             │
│     Prompt expansion (LLM rewrites for quality, like DALL-E 3)   │
│     Negative prompt injection (common quality improvements)       │
│     Language detection + translation to English if needed         │
│                                                                   │
│  3. GENERATION CLUSTER                                            │
│     GPU pool: 20-40 A100/H100 GPUs                                │
│     Model: SDXL/Flux with LoRA adapters for styles               │
│     Optimization: TensorRT, fp16, 20-step DDIM (not 50 DDPM)    │
│     Batch: accumulate requests, batch-generate (4-8 per batch)   │
│     Target: ~2s per image on A100 with optimizations              │
│                                                                   │
│  4. POST-PROCESSING                                               │
│     NSFW classifier on output (filter unsafe generations)         │
│     Upscaling: Real-ESRGAN for 4× resolution boost               │
│     Watermarking: invisible watermark in generated images         │
│     CDN upload: S3/CloudFront for fast delivery                   │
│                                                                   │
│  5. CACHING & OPTIMIZATION                                        │
│     Semantic prompt cache: similar prompts → cached results       │
│     Popular styles pre-generated (daily trending)                  │
│     Prompt embedding cache (avoid re-encoding CLIP)               │
│                                                                   │
│  6. COST OPTIMIZATION                                             │
│     Spot/preemptible GPUs for non-urgent requests                │
│     Model distillation: 4-step consistency model for drafts      │
│     Tiered quality: fast/draft (4 steps) vs high (20 steps)      │
│     ~$0.02-0.04 per image at scale                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 14.11 Key Papers

| Paper                                                                                  | Year | Why It Matters                               |
| -------------------------------------------------------------------------------------- | ---- | -------------------------------------------- |
| _Denoising Diffusion Probabilistic Models_ (Ho et al.)                                 | 2020 | DDPM — modern diffusion models               |
| _High-Resolution Image Synthesis with Latent Diffusion_ (Rombach et al.)               | 2022 | Stable Diffusion — latent space diffusion    |
| _Learning Transferable Visual Models from NL Supervision_ (Radford et al.)             | 2021 | CLIP — connected vision and language         |
| _Scalable Diffusion Models with Transformers_ (Peebles & Xie)                          | 2023 | DiT — transformers replace U-Net             |
| _Classifier-Free Diffusion Guidance_ (Ho & Salimans)                                   | 2022 | CFG — the key to text-guided generation      |
| _Adding Conditional Control to Text-to-Image Diffusion_ (Zhang et al.)                 | 2023 | ControlNet — spatial control over generation |
| _Denoising Diffusion Implicit Models_ (Song et al.)                                    | 2020 | DDIM — faster sampling (10-50× speedup)      |
| _Video Generation Models as World Simulators_ (OpenAI)                                 | 2024 | Sora technical report — video DiT            |
| _Robust Speech Recognition via Large-Scale Weak Supervision_ (Radford et al.)          | 2022 | Whisper — universal speech recognition       |
| _Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers_ (Wang et al.) | 2023 | VALL-E — LLM approach to speech synthesis    |

---

[← Module 13: Deep Learning](../module-13-deep-learning/README.md) | [Module 15: MLOps →](../module-15-mlops/README.md)
