from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = pd.read_csv("data/country_business_data.csv")
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data["business_index"], kde=True, ax=ax)
    ax.set_title("Distribution of business index")
    fig.tight_layout()
    fig.savefig(out / "eda_target_distribution.png", dpi=140)
    plt.close(fig)

    numeric = data.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation between numeric features")
    fig.tight_layout()
    fig.savefig(out / "eda_correlation.png", dpi=140)
    plt.close(fig)
    print(f"EDA charts saved to {out.resolve()}")
