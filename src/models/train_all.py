"""Train both GNN models in sequence.

  1. ASSOCIATED_WITH  — the real, headline model (trustworthy metrics)
  2. EXPLOITS          — the experimental model (heuristic edges; caveated)

Run from project root (after build_features):
    python -m src.models.train_all
"""
from src.models.train import train


def main():
    print("#" * 60)
    print("# MODEL 1/2 — ASSOCIATED_WITH (real labels, headline result)")
    print("#" * 60)
    train(target="ASSOCIATED_WITH", epochs=200)

    print("\n" + "#" * 60)
    print("# MODEL 2/2 — EXPLOITS (heuristic edges, experimental)")
    print("#" * 60)
    train(target="EXPLOITS", sources="all", epochs=200)


if __name__ == "__main__":
    main()
