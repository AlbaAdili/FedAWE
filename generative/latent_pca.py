import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from vae_model import VAE, set_flat_params
from mnist_loader import get_dataset, make_test_loader


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VAE(ckpt.get("latent_dim", 20)).to(device)
    set_flat_params(model, ckpt["weights"], device)
    model.eval()
    return model, ckpt


def collect_latents(model, loader, device, max_points=2000):
    mus = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, _ = model.encode(x)

            mus.append(mu.cpu())
            labels.append(y.cpu())

            if sum(len(t) for t in labels) >= max_points:
                break

    mus = torch.cat(mus, dim=0)[:max_points].numpy()
    labels = torch.cat(labels, dim=0)[:max_points].numpy()

    return mus, labels


def plot_pca(checkpoints, names, dataset, output_path, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    _, test_data = get_dataset(dataset)
    test_loader = make_test_loader(test_data)

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 4))

    if len(checkpoints) == 1:
        axes = [axes]

    for ax, checkpoint, name in zip(axes, checkpoints, names):
        model, _ = load_model(checkpoint, device)

        z, labels = collect_latents(model, test_loader, device)

        z_2d = PCA(n_components=2).fit_transform(z)

        scatter = ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=labels,
            s=6,
            alpha=0.65,
            cmap="tab10"
        )

        ax.set_title(name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.colorbar(scatter, ax=axes, label="Class label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved PCA plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs("generative_results/latent", exist_ok=True)

    checkpoints = [
        "generative_results/checkpoints/fedavg_vae_mnist_high_dropout_dirichlet_seed3.pt",
        "generative_results/checkpoints/fedawe_vae_mnist_high_dropout_dirichlet_seed3.pt",
        "generative_results/checkpoints/selective_echo_fedavg_vae_mnist_high_dropout_dirichlet_seed3_stale5.pt",
    ]

    names = [
        "FedAvg-VAE",
        "FedAWE-VAE",
        "Selective Echo",
    ]

    output_path = "generative_results/latent/latent_pca_mnist_high_dropout.png"

    plot_pca(
        checkpoints=checkpoints,
        names=names,
        dataset=args.dataset,
        output_path=output_path,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()