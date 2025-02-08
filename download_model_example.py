from transformers import AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model downloaded successfully!")
