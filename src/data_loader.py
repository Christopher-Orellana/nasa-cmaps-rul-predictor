from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_fd001(
        data_path: str | Path,
        rul_col: str = 'RUL',
        cap_rul: str = 125,
) -> pd.DataFrame:
    """ Load the processed FD001 dataset and apply the shared preprocessing"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Data file was not found: {data_path.resolve()}')

    df = pd.read_csv(data_path)

    if rul_col not in df.columns:
        raise KeyError(
            f'Expected column "{rul_col}" not found. '
            f'Available columns: {list(df.columns)}'
        )

    # Shared target for modeling
    df['RUL_capped'] = df[rul_col].clip(upper=cap_rul)