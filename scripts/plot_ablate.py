import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ablate(title, path, out_path):
    df = pd.read_csv(path)
    df["eval"] = df["eval"].apply(lambda x: x.split("/")[-1])
    fig, ax = plt.subplots(figsize=(20, 3))
    sns.barplot(df, x="eval", y="score", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("layer")

    if "inf" not in out_path:
        fig.savefig(out_path, dpi=300)
        return

    for i, bar in enumerate(ax.patches):
        r2_val = df["inf_r2"].iloc[i]
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"R²={r2_val:.2f}",
            ha="center",
            va="bottom",
            rotation=45,  # <-- rotate text
            rotation_mode="anchor",
        )

    fig.savefig(out_path, dpi=300)


OUTPUT_DIR = "inference/fadtk"

plot_ablate(r"$FAD_{clap}$ YuE fma_pop", f"{OUTPUT_DIR}/fad.csv", f"{OUTPUT_DIR}/fad.png")
plot_ablate(r"$FAD_{\infty}$ YuE fma_pop", f"{OUTPUT_DIR}/fad.csv", f"{OUTPUT_DIR}/fad_inf.png")
plot_ablate(r"$FAD_{clap}$ YuE pure relative", f"{OUTPUT_DIR}/fad-relative.csv", f"{OUTPUT_DIR}/fad_relative.png")
plot_ablate(r"$FAD_{\infty}$ YuE pure relative", f"{OUTPUT_DIR}/fad-relative.csv", f"{OUTPUT_DIR}/fad_relative_inf.png")
