from __future__ import annotations
import pandas as pd
from ucimlrepo import fetch_ucirepo
from .utils import RAW_DIR
OUTPUT_PATH = RAW_DIR / "online_retail_ii.csv"
def main() -> None:
    dataset = fetch_ucirepo(id=502)
    df = dataset.data.original if hasattr(dataset.data, "original") else pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved raw data to {OUTPUT_PATH}")
if __name__ == "__main__":
    main()
