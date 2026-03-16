"""Download torchvision ViT-B/16 pretrained weights into torch cache."""

from __future__ import annotations

from torchvision.models import ViT_B_16_Weights, vit_b_16


def main() -> None:
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    n = sum(p.numel() for p in model.parameters())
    print(f"Downloaded/loaded ViT-B/16 weights. Parameters: {n:,}")


if __name__ == "__main__":
    main()
