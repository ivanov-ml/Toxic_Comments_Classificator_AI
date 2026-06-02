from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')

df = df.dropna(subset=['comment_text'])

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df['comment_text']
y = df[label_cols]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = MultiOutputClassifier(LogisticRegression(class_weight='balanced'))
model.fit(X_train_vec, y_train)

y_pred = model.predict_proba(X_val_vec)


auc_scores = []
for i, col in enumerate(label_cols):
    auc = roc_auc_score(y_val[col], y_pred[i][:, 1])
    print(f"{col}: {auc:.4f}")
    auc_scores.append(auc)

print(f"Mean ROC-AUC: {sum(auc_scores) / len(auc_scores):.4f}")


