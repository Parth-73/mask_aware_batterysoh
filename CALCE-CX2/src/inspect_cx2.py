
from pathlib import Path
import pandas as pd

root = Path(r"D:\battery_soh\CALCE-CX2\CX2_data")

files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in [".xlsx", ".xls", ".txt"]
)

print("Total data files:", len(files))

# Inspect first Excel file
excel_files = [
    p for p in files if p.suffix.lower() in [".xlsx", ".xls"]
]

if not excel_files:
    raise RuntimeError("No Excel files found")

file = excel_files[0]

print("\nBattery file:", file.name)

book = pd.ExcelFile(file)

print("\nAvailable sheets:")
for sheet in book.sheet_names:
    print(" -", sheet)

for sheet in book.sheet_names:
    if "Channel" in sheet:
        df = pd.read_excel(file, sheet_name=sheet, nrows=5)

        print("\nSelected sheet:", sheet)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nSample rows:")
        print(df.to_string(index=False))
        break
