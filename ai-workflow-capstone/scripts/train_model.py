from pathlib import Path
import pandas as pd
from app.model import train_and_evaluate

if __name__ == "__main__":
    data_path = Path("data/country_business_data.csv")
    if not data_path.exists():
        raise SystemExit(f"Dataset missing: {data_path}")
    result = train_and_evaluate(pd.read_csv(data_path))
    print(f"Selected model: {result['selected_model']}")
    for name, metrics in result["models"].items():
        print(f"{name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
