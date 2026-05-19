from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL
# =====================================================

class TextClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "bert-base-uncased"
        )

        hidden = self.bert.config.hidden_size

        self.classifier = nn.Sequential(

            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)

        )

    def forward(self, input_ids, attention_mask):

        output = self.bert(

            input_ids=input_ids,
            attention_mask=attention_mask

        )

        cls_embedding = output.last_hidden_state[:, 0, :]

        return self.classifier(cls_embedding)

# =====================================================
# LOAD TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = TextClassifier()

model.load_state_dict(

    torch.load(
        "saved_models/text_classifier.pth",
        map_location=DEVICE
    )

)

model.to(DEVICE)

model.eval()

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_news(text):

    encoded = tokenizer(

        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"

    )

    with torch.no_grad():

        outputs = model(

            encoded["input_ids"].to(DEVICE),
            encoded["attention_mask"].to(DEVICE)

        )

    probs = F.softmax(outputs, dim=1)

    prediction = probs.argmax(dim=1).item()

    confidence = probs[0][prediction].item() * 100

    if prediction == 0:

        label = "FAKE NEWS ❌"

    else:

        label = "REAL NEWS ✅"

    return label, round(confidence, 2)

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        news = request.form["news"]

        prediction, confidence = predict_news(news)

    return render_template(

        "index.html",
        prediction=prediction,
        confidence=confidence

    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)