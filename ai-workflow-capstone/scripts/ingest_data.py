"""Validate and normalize the input CSV; can be scheduled in an automation workflow."""
from pathlib import Path
import pandas as pd

SOURCE = Path("data/country_business_data.csv")
OUTPUT = Path("data/country_business_data.csv")

def ingest(source=SOURCE, output=OUTPUT):
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    if "country" not in df.columns:
        raise ValueError("Input CSV must include a country column")
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"] != ""].drop_duplicates(subset=["country"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Ingested {len(df)} country rows into {output}")
    return df

if __name__ == "__main__":
    ingest()
