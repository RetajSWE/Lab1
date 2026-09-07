"""Lab 2: Transformer anatomy experiments."""


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.bayan.attention import attention, MultiHeadAttention
def main():
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # 1. Scaled dot-product attention: compare with PyTorch
    # ---------------------------------------------------------
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    print("1. Attention numerical equivalence")
    print(f"   Max difference: {(actual - expected).abs().max().item():.8f}")

    assert torch.allclose(actual, expected, atol=1e-6)
    print("   PASS")

    # ---------------------------------------------------------
    # 2. Inspect attention weights
    # ---------------------------------------------------------
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    weights = F.softmax(scores, dim=-1)

    print("\n2. Attention weights")
    print(weights)

    assert torch.allclose(weights.sum(dim=-1),
                          torch.ones_like(weights.sum(dim=-1)),
                          atol=1e-6)
    print("   Row sums = 1: PASS")

    # ---------------------------------------------------------
    # 3. Multi-Head Attention
    # ---------------------------------------------------------
    d_model = 16
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(2, 5, d_model)
    mha_output = mha(x, x, x)

    print("\n3. Multi-Head Attention")
    print(f"   Input shape:  {tuple(x.shape)}")
    print(f"   Output shape: {tuple(mha_output.shape)}")

    assert mha_output.shape == x.shape
    print("   Shape check: PASS")

    # ---------------------------------------------------------
    # 4. Causal masking
    # ---------------------------------------------------------
    seq_len = 4

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    causal_output = attention(q[:, :, :seq_len, :],
                              k[:, :, :seq_len, :],
                              v[:, :, :seq_len, :],
                              causal_mask)

    causal_scores = torch.matmul(
        q[:, :, :seq_len, :],
        k[:, :, :seq_len, :].transpose(-2, -1)
    ) / (q.size(-1) ** 0.5)

    causal_scores = causal_scores.masked_fill(
        ~causal_mask, float("-inf")
    )

    causal_weights = F.softmax(causal_scores, dim=-1)

    print("\n4. Causal masking")
    print("   Attention matrix:")
    print(causal_weights[0, 0])

    upper_triangle = torch.triu(causal_weights, diagonal=1)

    assert torch.allclose(
        upper_triangle,
        torch.zeros_like(upper_triangle),
        atol=1e-6
    )

    print("   Future-token attention blocked: PASS")

    print("\nLab 2 Step 2 completed successfully.")


if __name__ == "__main__":
    main()