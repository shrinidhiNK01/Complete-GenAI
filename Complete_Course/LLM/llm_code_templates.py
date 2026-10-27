"""
LLM Code Templates

This file contains small, educational code patterns for:
- gradient descent
- backpropagation
- tokenization
- next-token prediction
- attention
- transformer blocks
- tiny GPT-style generation
- retrieval augmented generation
- sampling
- toy diffusion

Install optional packages if needed:
    pip install numpy torch scikit-learn

The NumPy examples are easiest to understand.
The PyTorch examples are closer to real LLM code.
"""


# ============================================================
# 1. Gradient Descent For y = w*x + b
# ============================================================

def gradient_descent_linear_regression_demo():
    import numpy as np

    # Data: y is roughly 2*x + 1
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)

    w = 0.0
    b = 0.0
    lr = 0.01

    for epoch in range(1000):
        y_hat = w * X + b
        loss = np.mean((y_hat - y) ** 2)

        # Gradients for mean squared error
        dloss_dyhat = 2 * (y_hat - y) / len(X)
        dloss_dw = np.sum(dloss_dyhat * X)
        dloss_db = np.sum(dloss_dyhat)

        w = w - lr * dloss_dw
        b = b - lr * dloss_db

        if epoch % 200 == 0:
            print(f"epoch={epoch}, loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    print("Final:", {"w": w, "b": b})
    return w, b


# ============================================================
# 2. Tiny Neural Network Backpropagation From Scratch
# ============================================================

def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tiny_neural_network_backprop_demo():
    import numpy as np

    # XOR data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([[0], [1], [1], [0]], dtype=float)

    rng = np.random.default_rng(42)

    # 2 inputs -> 2 hidden neurons -> 1 output neuron
    W1 = rng.normal(size=(2, 2)) * 0.5
    b1 = np.zeros((1, 2))
    W2 = rng.normal(size=(2, 1)) * 0.5
    b2 = np.zeros((1, 1))

    lr = 0.5

    for epoch in range(10000):
        # Forward pass
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_hat = sigmoid(z2)

        loss = np.mean((y_hat - y) ** 2)

        # Backward pass
        dloss_dyhat = 2 * (y_hat - y) / len(X)
        dyhat_dz2 = sigmoid_derivative(z2)
        dloss_dz2 = dloss_dyhat * dyhat_dz2

        dloss_dW2 = a1.T @ dloss_dz2
        dloss_db2 = np.sum(dloss_dz2, axis=0, keepdims=True)

        dloss_da1 = dloss_dz2 @ W2.T
        da1_dz1 = sigmoid_derivative(z1)
        dloss_dz1 = dloss_da1 * da1_dz1

        dloss_dW1 = X.T @ dloss_dz1
        dloss_db1 = np.sum(dloss_dz1, axis=0, keepdims=True)

        # Gradient descent update
        W2 -= lr * dloss_dW2
        b2 -= lr * dloss_db2
        W1 -= lr * dloss_dW1
        b1 -= lr * dloss_db1

        if epoch % 2000 == 0:
            print(f"epoch={epoch}, loss={loss:.6f}")

    print("Predictions:")
    print(y_hat.round(3))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ============================================================
# 3. Simple Character Tokenizer
# ============================================================

class CharacterTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def character_tokenizer_demo():
    text = "hello language model"
    tokenizer = CharacterTokenizer(text)
    ids = tokenizer.encode("hello")
    print("Token ids:", ids)
    print("Decoded:", tokenizer.decode(ids))
    print("Vocab size:", tokenizer.vocab_size)
    return tokenizer


# ============================================================
# 4. Softmax, Cross Entropy, And Sampling
# ============================================================

def softmax(logits):
    import numpy as np

    logits = np.array(logits, dtype=float)
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


def cross_entropy_loss(probabilities, target_index):
    import numpy as np

    eps = 1e-12
    return -np.log(probabilities[target_index] + eps)


def sample_from_logits(logits, temperature=1.0, top_k=None):
    import numpy as np

    logits = np.array(logits, dtype=float) / temperature

    if top_k is not None:
        top_indices = np.argsort(logits)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[top_indices] = logits[top_indices]
        logits = filtered

    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


def softmax_sampling_demo():
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    print("Probabilities:", probs)
    print("Cross entropy if target is class 0:", cross_entropy_loss(probs, 0))
    print("Sampled token id:", sample_from_logits(logits, temperature=0.8, top_k=2))


# ============================================================
# 5. Scaled Dot-Product Attention In NumPy
# ============================================================

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    import numpy as np

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    output = weights @ V
    return output, weights


def attention_numpy_demo():
    import numpy as np

    # 3 tokens, vector size 4
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0]
    ])

    rng = np.random.default_rng(1)
    Wq = rng.normal(size=(4, 4))
    Wk = rng.normal(size=(4, 4))
    Wv = rng.normal(size=(4, 4))

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    output, weights = scaled_dot_product_attention_numpy(Q, K, V)
    print("Attention weights:")
    print(weights.round(3))
    print("Attention output:")
    print(output.round(3))
    return output, weights


# ============================================================
# 6. Causal Mask For Decoder-Only Models
# ============================================================

def causal_mask(size):
    import numpy as np
    return np.tril(np.ones((size, size), dtype=int))


def causal_attention_demo():
    import numpy as np

    X = np.eye(4)
    mask = causal_mask(4)
    output, weights = scaled_dot_product_attention_numpy(X, X, X, mask=mask)
    print("Causal mask:")
    print(mask)
    print("Causal attention weights:")
    print(weights.round(3))
    return output, weights


# ============================================================
# 7. Bigram Language Model In PyTorch
# ============================================================

def train_bigram_language_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    text = (
        "hello world. this is a tiny language model. "
        "language models predict the next token. "
    )

    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):
            logits = self.token_embedding_table(idx)

            if targets is None:
                loss = None
            else:
                batch, time, channels = logits.shape
                logits_flat = logits.view(batch * time, channels)
                targets_flat = targets.view(batch * time)
                loss = F.cross_entropy(logits_flat, targets_flat)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                logits, _ = self(idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_idx], dim=1)
            return idx

    def get_batch(batch_size=8, block_size=16):
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
        return x, y

    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for step in range(500):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("step", step, "loss", float(loss))

    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=80)[0].tolist()
    print(tokenizer.decode(generated))
    return model, tokenizer


# ============================================================
# 8. Transformer Components In PyTorch
# ============================================================

def transformer_components_demo():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
            super().__init__()
            assert embed_dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

        def forward(self, x):
            batch, time, channels = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

            scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            out = weights @ v
            out = out.transpose(1, 2).contiguous().view(batch, time, channels)
            return self.proj(out)

    class FeedForward(nn.Module):
        def __init__(self, embed_dim, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = MultiHeadSelfAttention(embed_dim, num_heads, block_size, dropout)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.ff = FeedForward(embed_dim, dropout)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    x = torch.randn(2, 8, 32)
    block = TransformerBlock(embed_dim=32, num_heads=4, block_size=8)
    y = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    return block


# ============================================================
# 9. Tiny GPT-Style Model
# ============================================================

def train_tiny_gpt_style_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    text = (
        "deep learning learns patterns from data. "
        "attention lets tokens talk to other tokens. "
        "transformers are powerful language models. "
        "large language models predict the next token. "
    )

    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    batch_size = 16
    block_size = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

        def forward(self, x):
            b, t, c = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            head_dim = c // num_heads

            q = q.view(b, t, num_heads, head_dim).transpose(1, 2)
            k = k.view(b, t, num_heads, head_dim).transpose(1, 2)
            v = v.view(b, t, num_heads, head_dim).transpose(1, 2)

            scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
            scores = scores.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = weights @ v
            out = out.transpose(1, 2).contiguous().view(b, t, c)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(block_size, embed_dim)
            self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
            self.ln_f = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, tokenizer.vocab_size)

        def forward(self, idx, targets=None):
            b, t = idx.shape
            token_emb = self.token_embedding(idx)
            pos = torch.arange(t, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = token_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(b * t, tokenizer.vocab_size),
                    targets.view(b * t)
                )

            return logits, loss

        def generate(self, idx, max_new_tokens, temperature=1.0):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_idx], dim=1)
            return idx

    def get_batch():
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
        return x, y

    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(300):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("step", step, "loss", float(loss))

    start_text = "deep"
    start = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long)
    generated = model.generate(start, max_new_tokens=120, temperature=0.8)[0].tolist()
    print(tokenizer.decode(generated))
    return model, tokenizer


# ============================================================
# 10. Simple RAG With TF-IDF
# ============================================================

def simple_rag_tfidf_demo():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [
        "Gradient descent updates weights in the direction that reduces loss.",
        "Backpropagation computes gradients using the chain rule.",
        "Transformers use attention to let tokens communicate with other tokens.",
        "RAG retrieves relevant documents and gives them to a language model as context.",
        "Diffusion models generate images by denoising random noise step by step."
    ]

    question = "How does backpropagation work?"

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, doc_vectors).ravel()
    top_index = similarities.argmax()
    retrieved_doc = documents[top_index]

    prompt = f"""
Use the context to answer the question.

Context:
{retrieved_doc}

Question:
{question}

Answer:
"""

    print("Retrieved document:", retrieved_doc)
    print("Prompt to send to LLM:")
    print(prompt)
    return prompt


# ============================================================
# 11. LoRA Concept Sketch
# ============================================================

def lora_linear_layer_sketch():
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, in_features, out_features, rank=4, alpha=1.0):
            super().__init__()
            self.base = nn.Linear(in_features, out_features)
            self.base.weight.requires_grad = False
            self.base.bias.requires_grad = False

            self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.B = nn.Parameter(torch.zeros(rank, out_features))
            self.scale = alpha / rank

        def forward(self, x):
            base_output = self.base(x)
            lora_update = x @ self.A @ self.B * self.scale
            return base_output + lora_update

    layer = LoRALinear(8, 4, rank=2)
    x = torch.randn(3, 8)
    y = layer(x)
    print("Output shape:", y.shape)
    print("Trainable parameters:", sum(p.numel() for p in layer.parameters() if p.requires_grad))
    return layer


# ============================================================
# 12. Toy Diffusion Denoising Example
# ============================================================

def toy_diffusion_1d_demo():
    import numpy as np

    # A tiny toy example, not a real image diffusion model.
    clean_signal = np.sin(np.linspace(0, 2 * np.pi, 100))

    noise_level = 0.8
    noisy_signal = clean_signal + noise_level * np.random.randn(100)

    # Simple hand-written "denoiser": moving average.
    # Real diffusion models learn a neural denoiser.
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    denoised_signal = np.convolve(noisy_signal, kernel, mode="same")

    print("Clean first 5:", clean_signal[:5].round(3))
    print("Noisy first 5:", noisy_signal[:5].round(3))
    print("Denoised first 5:", denoised_signal[:5].round(3))

    return clean_signal, noisy_signal, denoised_signal


if __name__ == "__main__":
    print("LLM code templates loaded.")
    print("Run individual functions to study each concept.")

