from flask import Flask, render_template_string, request
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

        cls_emb = output.last_hidden_state[:, 0, :]

        return self.classifier(cls_emb)

# =====================================================
# LOAD TOKENIZER
# =====================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = TextClassifier()

model.load_state_dict(

    torch.load(
        "saved_models/text_classifier.pth",
        map_location=DEVICE
    )

)

model.to(DEVICE)

model.eval()

print("Model loaded successfully!")

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_news(text):

    enc = tokenizer(

        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"

    )

    with torch.no_grad():

        logits = model(

            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)

        )

    probs = F.softmax(logits, dim=1)

    idx = probs.argmax(dim=1).item()

    confidence = probs[0][idx].item() * 100

    label = "FAKE ❌" if idx == 0 else "REAL ✅"

    return label, round(confidence, 2)

# =====================================================
# HTML DASHBOARD
# =====================================================

HTML = """

<!DOCTYPE html>
<html>

<head>

    <title>Fake News Detection AI</title>

    <style>

        body {

            margin: 0;
            padding: 0;
            font-family: Arial;
            background: linear-gradient(135deg,#141e30,#243b55);
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;

        }

        .container {

            width: 700px;
            background: rgba(255,255,255,0.1);
            padding: 40px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 20px rgba(0,0,0,0.4);

        }

        h1 {

            text-align: center;
            margin-bottom: 30px;
            font-size: 38px;

        }

        textarea {

            width: 100%;
            height: 180px;
            border: none;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            resize: none;
            outline: none;

        }

        button {

            margin-top: 20px;
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            background: #00c6ff;
            color: white;
            cursor: pointer;
            transition: 0.3s;

        }

        button:hover {

            background: #0072ff;

        }

        .result {

            margin-top: 30px;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            background: rgba(255,255,255,0.15);

        }

        .label {

            font-size: 32px;
            font-weight: bold;

        }

        .confidence {

            margin-top: 10px;
            font-size: 22px;

        }

    </style>

</head>

<body>

    <div class="container">

        <h1>📰 Fake News Detection AI</h1>

        <form method="POST">

            <textarea
                name="news"
                placeholder="Paste your news text here..."
                required
            ></textarea>

            <button type="submit">
                Detect News
            </button>

        </form>

        {% if prediction %}

        <div class="result">

            <div class="label">
                {{ prediction }}
            </div>

            <div class="confidence">
                Confidence : {{ confidence }}%
            </div>

        </div>

        {% endif %}

    </div>

</body>
</html>

"""

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        text = request.form["news"]

        prediction, confidence = predict_news(text)

    return render_template_string(

        HTML,
        prediction=prediction,
        confidence=confidence

    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(

        debug=True,
        host="0.0.0.0",
        port=5000

    )