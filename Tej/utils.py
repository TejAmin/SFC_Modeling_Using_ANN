from pathlib import Path
import pandas as pd
from typing import Union, Dict, List


def load_txt_folder(
    folder_path: Union[str, Path],
    sep: str = r"\s+",
    concat: bool = True,
    add_source_col: bool = True,
    **read_csv_kw,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Read every *.txt file in *folder_path*.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Directory containing the files.
    sep : str, default r"\\s+"
        Delimiter passed to pandas.read_csv.
    concat : bool, default True
        If True, return one big DataFrame; otherwise return a dict.
    add_source_col : bool, default True
        When concat=True, add a 'source' column with the file-stem.
    **read_csv_kw
        Extra keyword args forwarded to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame  or  dict[str, pandas.DataFrame]
    """
    folder = Path(folder_path).expanduser().resolve()
    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder}")

    if concat:
        frames: List[pd.DataFrame] = []
        for f in txt_files:
            df = pd.read_csv(f, sep=sep, **read_csv_kw)
            if add_source_col:
                df["source"] = f.stem
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # keep each DataFrame separate
    return {f.stem: pd.read_csv(f, sep=sep, **read_csv_kw) for f in txt_files}


