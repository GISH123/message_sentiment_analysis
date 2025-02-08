import os
import pandas as pd
import torch
from transformers import pipeline

# Ensure PyTorch is using GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Directories for input CSVs and output
data_dir = "raw_data"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# 11 models that can (in principle) do sentiment analysis
# NOTE: If any fail in practice, remove or replace them with a truly fine-tuned model.
model_list = {
    "bert_multilingual":     "nlptown/bert-base-multilingual-uncased-sentiment",   # 1-5 star
    "RoBERTa_tl":           "dost-asti/RoBERTa-tl-sentiment-analysis",            # 1-5 star
    "Multilingual_SA":      "tabularisai/multilingual-sentiment-analysis",        # neg/neu/pos
    "RoBERTa_tl_large":     "jcblaise/roberta-tagalog-large",                     # Tagalog
    "BERT_multilingual":    "bert-base-multilingual-uncased",                     # may fail if not fine-tuned
    "XLM_RoBERTa_base":     "xlm-roberta-base",                                   # may fail if not fine-tuned
    "XLM_RoBERTa_large":    "xlm-roberta-large",                                  # may fail if not fine-tuned
    "mBERT":                "bert-base-multilingual-cased",                       # may fail if not fine-tuned
    "DistilBERT_multi":     "distilbert-base-multilingual-cased",                # may fail if not fine-tuned
    "XLMR_large_xnli":      "joeddav/xlm-roberta-large-xnli",                     # zero-shot => pos/neg/neutral
    "twitter_roberta":      "cardiffnlp/twitter-roberta-base-sentiment-latest",   # pos/neg/neu
}

##############################################################################
# Build a dictionary of pipelines
##############################################################################
pipelines_dict = {}
for model_name, model_path in model_list.items():
    print(f"📌 Loading model: {model_name} => {model_path}")
    try:
        pl = pipeline(
            "sentiment-analysis",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=16
        )
        pipelines_dict[model_name] = pl
    except Exception as e:
        print(f"❌ [ERROR] Could not load model {model_name}: {e}")
        # We'll store None so we can skip it at inference time
        pipelines_dict[model_name] = None

##############################################################################
# A helper to convert model output => star rating => negative score
##############################################################################
def get_negative_score(sentiment_result):
    """
    Convert the model's output (like '4 stars', 'positive', 'neutral', etc.)
    into a standardized negative emotion score (0-100).
    """
    label = sentiment_result["label"]
    score = sentiment_result["score"]

    # Case 1: 'X stars' => star rating
    if "star" in label.lower():
        # e.g. '5 stars' => star_rating = 5
        star_rating = label[0]  # just take the first digit
    # Case 2: label-based => 'positive', 'negative', 'neutral'
    elif label.lower() in ["negative", "neg"]:
        star_rating = 1
    elif label.lower() in ["neutral"]:
        star_rating = 3
    elif label.lower() in ["positive", "pos"]:
        star_rating = 5
    # Some zero-shot classification outputs => 'LABEL_0', 'LABEL_1', ...
    elif label.upper().startswith("LABEL_"):
        # heuristics: we might read the number and map it
        # or if the model returns 'LABEL_2' for positive, etc.
        # We'll assume LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
        idx = int(label.split("_")[-1])
        # map: 0=neg => 1 star, 1=neu => 3 stars, 2=pos => 5 stars
        star_rating_map = {0: 1, 1: 3, 2: 5}
        star_rating = star_rating_map.get(idx, 3)  # default to 3 if unknown
    else:
        # unrecognized => skip
        return None, None

    # Then compute negative intensity: 
    # (5 - star_rating) => how negative from 1..4
    # multiply by 'score' => confidence
    negative_intensity = (5 - star_rating) * score

    # standardize to 0..100
    negative_emotion = int((negative_intensity / 4.0) * 100)

    return label, negative_emotion

##############################################################################
# Main loop for CSV processing
##############################################################################
def process_csv(csv_path, output_path):
    """Analyze each row's message with multiple sentiment models and store results."""
    print(f"\n📂 Processing file: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "MESSAGE" not in df.columns:
        print(f"❌ Skipping {csv_path}: No 'MESSAGE' column found.")
        return

    messages = df["MESSAGE"].astype(str).tolist()

    for model_name, sentiment_pl in pipelines_dict.items():
        if sentiment_pl is None:
            # Model failed to load
            df[f"{model_name}_model_scores"] = None
            df[f"{model_name}_neg_score_standard"] = None
            continue

        print(f"   🧠 Running sentiment analysis with model: {model_name} ...")
        try:
            # 1) Run pipeline in batches
            sentiment_results = sentiment_pl(messages)

            # 2) Convert each result
            star_ratings, neg_scores = [], []
            for sres in sentiment_results:
                srating, nscore = get_negative_score(sres)
                star_ratings.append(srating)
                neg_scores.append(nscore)

            # 3) Add columns to DataFrame
            df[f"{model_name}_model_scores"] = star_ratings
            df[f"{model_name}_neg_score_standard"] = neg_scores

        except Exception as e:
            print(f"   ❌ Error with model {model_name}: {e}")
            df[f"{model_name}_model_scores"] = None
            df[f"{model_name}_neg_score_standard"] = None

    # Save CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Finished {csv_path} => {output_path}")

##############################################################################
# Process all CSV files in data_dir
##############################################################################
for filename in os.listdir(data_dir):
    if filename.lower().endswith(".csv"):
        csv_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        process_csv(csv_path, output_path)

print("🎯 **Sentiment analysis complete! See output CSVs in:**", output_dir)
