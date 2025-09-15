import gradio as gr
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from pathlib import Path

# ------------------------
# Load model & tokenizer
# ------------------------
MODEL_DIR = Path(__file__).parent / "bert_pair_classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device).eval()

# ------------------------
# Single Prediction
# ------------------------
def predict_single(feedback, reason):
    inputs = tokenizer(feedback, reason, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    label = "Aligned " if pred == 1 else "Not Aligned ❌"
    return label, {
        "Not Aligned (0)": round(float(probs[0]), 3),
        "Aligned (1)": round(float(probs[1]), 3)
    }

# ------------------------
# Batch Prediction from CSV
# ------------------------
def predict_csv(file):
    df = pd.read_csv(file.name)  # <-- Fix for Gradio File object
    assert {"text", "reason"}.issubset(df.columns), "CSV must have 'text' and 'reason' columns"

    results = []
    for _, row in df.iterrows():
        text, reason = row["text"], row["reason"]
        inputs = tokenizer(text, reason, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        results.append({
            "text": text,
            "reason": reason,
            "prediction": "Aligned " if pred == 1 else "Not Aligned ❌",
            "conf_0": round(float(probs[0]), 3),
            "conf_1": round(float(probs[1]), 3)
        })

    out_df = pd.DataFrame(results)
    output_file = "batch_predictions.csv"
    out_df.to_csv(output_file, index=False)
    return output_file

# ------------------------
# Gradio UI
# ------------------------
with gr.Blocks() as demo:
    gr.Markdown("##  My Zoom Feedback Validator")
    gr.Markdown("Validate a single feedback, or run batch predictions with a CSV.")

    with gr.Tab(" Single Prediction"):
        inp1 = gr.Textbox(label="Feedback Text", lines=3, placeholder="Enter student's feedback...")
        inp2 = gr.Textbox(label="Dropdown Reason", lines=2, placeholder="Enter reason selected from dropdown...")
        out1 = gr.Label(label="Prediction")
        out2 = gr.Label(label="Confidence Scores")
        btn1 = gr.Button("Predict")
        btn1.click(fn=predict_single, inputs=[inp1, inp2], outputs=[out1, out2])

    with gr.Tab(" Batch Prediction"):
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
        csv_output = gr.File(label="Download Predictions")
        btn2 = gr.Button("Run Batch Prediction")
        btn2.click(fn=predict_csv, inputs=csv_input, outputs=csv_output)

# ------------------------
# Launch
# ------------------------
if __name__ == "__main__":
    demo.launch()



