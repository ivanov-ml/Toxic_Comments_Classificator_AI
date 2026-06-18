from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib




df = pd.read_csv('data/cleaned_data.csv')
df = df.dropna(subset=['comment_text'])

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df['comment_text']
y = df[label_cols]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Pipeline: векторизация → классификация
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('clf', MultiOutputClassifier(LogisticRegression(class_weight='balanced')))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict_proba(X_val)


auc_scores = []
for i, col in enumerate(label_cols):
    auc = roc_auc_score(y_val[col], y_pred[i][:, 1])
    print(f"{col}: {auc:.4f}")
    auc_scores.append(auc)

print(f"Mean ROC-AUC: {sum(auc_scores) / len(auc_scores):.4f}")

joblib.dump(pipeline, 'app/models/toxic_pipeline.pkl')