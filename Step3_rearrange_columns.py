import os
import pandas as pd

# ================================================
# Configuration
# ================================================
data_dir = "translated_data_example"  # Folder containing CSV files
output_dir = "rearranged_data_example"  # Folder to save rearranged CSVs
os.makedirs(output_dir, exist_ok=True)

# ================================================
# Rearranging Columns Function
# ================================================

def rearrange_csv_columns(csv_path, output_path):
    """
    Rearranges the CSV so that 'translated_message_en' and 'translated_message_zh' 
    come right before the 'MESSAGE' column while keeping all other columns unchanged.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"[INFO] Processing file: {csv_path}")
    
    # Check if required columns exist
    if "MESSAGE" not in df.columns or "translated_message_en" not in df.columns or "translated_message_zh" not in df.columns:
        print("  [WARNING] Required columns missing. Skipping...")
        return
    
    # Get the column order
    columns = list(df.columns)
    
    # Move translated columns before MESSAGE
    columns.remove("translated_message_en")
    columns.remove("translated_message_zh")
    columns.remove("MESSAGE")
    new_column_order = columns[:columns.index("LOGINNAME")] + ["translated_message_en", "translated_message_zh", "MESSAGE"] + columns[columns.index("LOGINNAME"):]
    
    # Rearrange and save the new CSV
    df = df[new_column_order]
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[DONE] Saved rearranged CSV ⇒ {output_path}")

# ================================================
# Process All CSVs
# ================================================
if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".csv"):
            csv_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, f"rearranged_{filename}")
            rearrange_csv_columns(csv_path, output_path)

    print("\n✅ All files processed. See output in:", output_dir)
