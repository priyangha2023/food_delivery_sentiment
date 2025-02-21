# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

# Sample dataset (Modify with real reviews)
data = {
    "text": [
        "The food was great and delivered on time!",
        "Terrible service, my order was late and cold",
        "Loved the experience, will order again!",
        "The delivery guy was rude, very disappointed"
    ],
    "label": [1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Train the model
vectorizer = TfidfVectorizer()
model = MultinomialNB()

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", model)
])

pipeline.fit(df["text"], df["label"])

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(pipeline, "model/nb_model.pkl")
print("✅ Model saved successfully at model/nb_model.pkl")
