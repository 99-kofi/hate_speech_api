from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import urllib.request
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn

# ===== Download Model If Not Exists =====
MODEL_URL = "https://drive.google.com/uc?export=download&id=14ii_QL1JoWMXQ3MvXszoTmCYbUYfQwRf
"  # 👈 REPLACE THIS
MODEL_PATH = "obala_lang_model.pt"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")

# ===== Define Model Class =====
class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        return self.output(self.dropout(hidden))

# ===== Load Model =====
model = BERTClassifier(n_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("tokenizer/")

app = FastAPI()
label_map = {0: "Neutral", 1: "Offensive", 2: "Hate"}

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer.encode_plus(
        data.text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        prediction = torch.argmax(outputs, dim=1).item()
    return {"label": label_map[prediction]}
