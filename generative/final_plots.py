import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load CSVs
# =========================

fedavg = pd.read_csv(
    "generative_results/fedavg_vae_mnist_high_dropout_dirichlet_seed3.csv"
)

fedawe = pd.read_csv(
    "generative_results/fedawe_vae_mnist_high_dropout_dirichlet_seed3.csv"
)

quality = pd.read_csv(
    "generative_results/fedawe_quality_vae_mnist_high_dropout_dirichlet_seed3.csv"
)

selective = pd.read_csv(
    "generative_results/selective_echo_fedavg_vae_mnist_high_dropout_dirichlet_seed3.csv"
)

# =========================
# Plot Loss Curves
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    fedavg["round"],
    fedavg["loss"],
    label="FedAvg-VAE"
)

plt.plot(
    fedawe["round"],
    fedawe["loss"],
    label="FedAWE-VAE"
)

plt.plot(
    quality["round"],
    quality["loss"],
    label="Quality Echo"
)

plt.plot(
    selective["round"],
    selective["loss"],
    label="Selective Echo"
)

plt.xlabel("Communication Round")
plt.ylabel("Loss")
plt.title("Federated VAE Training under High Dropout")
plt.legend()

plt.savefig(
    "generative_results/final_loss_plot.png",
    bbox_inches="tight"
)

print("Saved final_loss_plot.png")

# =========================
# Diversity Bar Plot
# =========================

methods = [
    "FedAvg",
    "FedAWE",
    "Quality Echo",
    "Selective Echo"
]

diversity = [
    0.1284,
    0.0820,
    0.1032,
    0.1133
]

plt.figure(figsize=(7,5))

plt.bar(methods, diversity)

plt.ylabel("Diversity Score")
plt.title("Generative Diversity under High Dropout")

plt.savefig(
    "generative_results/final_diversity_plot.png",
    bbox_inches="tight"
)

print("Saved final_diversity_plot.png")