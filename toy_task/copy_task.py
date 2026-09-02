"""
Toy copy-task: validate that the from-scratch attention implementation actually learns.

Task: given a sequence of random integers (embedded), predict the same sequence back.
A model that truly learns attention will memorize positional correspondence and converge
fast. A buggy attention (wrong masking, wrong scaling, NaN issues) will fail to converge.

Run: python -m toy_task.copy_task
Expected: loss should drop below 0.01 within ~30 epochs.

Why this matters before Day 7: if the attention is broken, you want to know on a
5-second toy task, not after a 10-hour transformer training run on BraTS.
"""

import torch
import torch.nn as nn
import sys
import os

# Allow running from project root: python -m toy_task.copy_task
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import MultiHeadAttention
from models.transformer_block import TransformerEncoder


class CopyTaskModel(nn.Module):
    """
    Tiny transformer for the copy task.
    Input: integer sequence → Embedding → TransformerEncoder → Linear → logits
    """

    def __init__(self, vocab_size=16, seq_len=10, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Learned positional embeddings (seq is short, so learnable is fine)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            ffn_dim=embed_dim * 4,
            n_layers=n_layers,
            dropout=0.0,
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """x: (B, S) integer token ids"""
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0)
        tokens = self.embed(x) + self.pos_emb(positions)
        out = self.encoder(tokens, attn_mask=None)  # no masking — attend everywhere
        return self.head(out)  # (B, S, vocab_size)


def run_copy_task(
    vocab_size=16,
    seq_len=10,
    batch_size=64,
    n_epochs=50,
    lr=3e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running copy-task on {device}\n")

    model = CopyTaskModel(vocab_size=vocab_size, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        # Generate random sequences each batch (no fixed dataset — always fresh)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = x  # copy task: output == input

        logits = model(x)  # (B, S, vocab_size)
        loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            # Measure accuracy (fraction of tokens predicted correctly)
            preds = logits.argmax(dim=-1)
            acc = (preds == target).float().mean().item()
            print(f"  epoch {epoch:3d}/{n_epochs}  loss {loss.item():.4f}  acc {acc:.3f}")

    print()
    if loss.item() < 0.05:
        print("✓ PASS — attention converged on copy task. Safe to proceed to 3D model.")
    else:
        print("✗ FAIL — attention did not converge. Debug before continuing.")
        print("  Common causes: wrong scale (sqrt(head_dim)), wrong mask convention,")
        print("  NaN in attention weights, missing residual connection.")

    return loss.item() < 0.05


if __name__ == "__main__":
    run_copy_task()
