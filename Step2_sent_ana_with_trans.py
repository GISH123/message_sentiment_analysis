import os
import time
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =======================================================
# Configuration
# =======================================================
# Read from the folder where sent_ana_multimodel.py stored output
data_dir = "processed_data_0207_11_models"

# Where to store final CSVs with added translations
output_dir = data_dir + "_translated"
os.makedirs(output_dir, exist_ok=True)

# Decide on source/target languages for NLLB
SRC_LANG = "tgl"       # Tagalog (source)
TGT_LANG_EN = "eng_Latn"
TGT_LANG_ZH = "zho_Hans"

# Select the translation model (Facebook NLLB-200)
model_name = "facebook/nllb-200-distilled-600M"

# =======================================================
# Device Setup
# =======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# =======================================================
# Load Translation Model & Tokenizer
# =======================================================
print(f"[INFO] Loading translation model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Optionally, use FP16 to reduce GPU memory usage
model.half()

# You can reduce max length for shorter texts
tokenizer.model_max_length = 128

# =======================================================
# Translation Helper
# =======================================================
def translate_text(texts, src_lang, tgt_lang, batch_size=16):
    """
    Translate a list of texts from `src_lang` to `tgt_lang` using NLLB-200.
    Returns a list of translated strings.
    """
    translations = []

    # ✅ Correctly set source language
    tokenizer.src_lang = src_lang

    # ✅ Ensure the correct target language token is retrieved
    if tgt_lang not in tokenizer.get_vocab():
        print(f"[ERROR] Target language token <<{tgt_lang}>> not found in tokenizer vocab!")
        return ["ERROR: Invalid target language"] * len(texts)

    tgt_lang_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    print(f"[INFO] Translating {len(texts)} messages from {src_lang} to {tgt_lang} (batch_size={batch_size})")


    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  → Batch {i // batch_size + 1} / {(len(texts) + batch_size - 1) // batch_size}")

        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.inference_mode():  # More efficient than torch.no_grad()
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_token_id,  # ✅ Ensure correct target language
                max_length=128
            )

        batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)

    total_time = time.time() - start_time
    print(f"[DONE] Translation completed in {total_time:.2f} seconds")

    return translations



# =======================================================
# Main Processing Function
# =======================================================
def process_csv(csv_path, output_path):
    """
    Reads a CSV with Tagalog text in 'MESSAGE' column (already containing
    multi-model sentiment results), translates each row to English & Chinese,
    and saves the final CSV with 2 new columns:
        'translated_message_en'
        'translated_message_zh'
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"\n[INFO] Processing file: {csv_path}")
    print(f"       Total rows: {len(df)}")

    # If for some reason your CSV doesn't have MESSAGE, adjust or skip
    if "MESSAGE" not in df.columns:
        print("  [WARNING] No 'MESSAGE' column found. Skipping...")
        return

    # Extract Tagalog messages
    messages_tl = df["MESSAGE"].fillna("").astype(str).tolist()

    # Translate to English
    print("\n[STEP] Translating Tagalog → English...")
    translated_en = translate_text(
        messages_tl, 
        src_lang=SRC_LANG, 
        tgt_lang=TGT_LANG_EN, 
        batch_size=16
    )

    # Translate to Chinese
    print("\n[STEP] Translating Tagalog → Chinese...")
    translated_zh = translate_text(
        messages_tl, 
        src_lang=SRC_LANG, 
        tgt_lang=TGT_LANG_ZH, 
        batch_size=16
    )

    # Add new columns
    df["translated_message_en"] = translated_en
    df["translated_message_zh"] = translated_zh

    # Save final CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[DONE] Saved translated CSV ⇒ {output_path}")


# =======================================================
# Run Over All CSVs in data_dir
# =======================================================
if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".csv"):
            csv_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, f"translated_{filename}")

            process_csv(csv_path, output_path)

    print("\n✅ All translations complete. See output in:", output_dir)
