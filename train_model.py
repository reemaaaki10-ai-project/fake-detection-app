import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL
# =========================================================

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

# =========================================================
# CREATE MODEL
# =========================================================

model = TextClassifier()

# =========================================================
# CREATE FOLDER
# =========================================================

os.makedirs("saved_models", exist_ok=True)

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(

    model.state_dict(),
    "saved_models/text_classifier.pth"

)

print("MODEL SAVED SUCCESSFULLY")