"""Build a small test_data/ subsample with 10 TICs from the full parquets."""
import os
import pandas as pd

OUT = "test_data"
os.makedirs(OUT, exist_ok=True)

peaks = pd.read_parquet("peaks.parquet")
tic_ids = sorted(peaks["TIC"].drop_duplicates().sample(10, random_state=42).tolist())
print("Selected TICs:", tic_ids)
pd.Series(tic_ids, name="TIC").to_csv(f"{OUT}/test_tics.csv", index=False)

def sub(path, out_name):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"SKIP {path}: {e}")
        return
    if "TIC" not in df.columns:
        print(f"SKIP {path}: no TIC column. cols={list(df.columns)}")
        return
    s = df[df["TIC"].isin(tic_ids)]
    s.to_parquet(f"{OUT}/{out_name}", index=False)
    print(f"{out_name}: {s.shape} -> {os.path.getsize(f'{OUT}/{out_name}')/1e6:.2f} MB")

sub("peaks.parquet", "peaks.parquet")
sub("peaks_ls.parquet", "peaks_ls.parquet")
sub("peaks_single.parquet", "peaks_single.parquet")
sub("lightcurves_all.parquet", "lightcurves_all.parquet")
sub("periodograms_acf.parquet", "periodograms_acf.parquet")
sub("periodograms_ls.parquet", "periodograms_ls.parquet")
