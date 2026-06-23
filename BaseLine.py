from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
import mlflow
import mlflow.sklearn



# Загрузка данных
df = pd.read_csv('data/cleaned_data.csv')
df = df.dropna(subset=['comment_text'])

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df['comment_text']
y = df[label_cols]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Параметры векторизатора и модели
tfidf_params = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'stop_words': 'english'
}
clf_params = {
    'class_weight': 'balanced'
}

# Устанавливаем tracking URI на запущенный сервер
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Запускаем MLflow эксперимент
with mlflow.start_run(run_name="toxic_comments_baseline"):
    # Логируем параметры
    mlflow.log_param("tfidf_max_features", tfidf_params['max_features'])
    mlflow.log_param("tfidf_ngram_range", tfidf_params['ngram_range'])
    mlflow.log_param("tfidf_stop_words", tfidf_params['stop_words'])
    mlflow.log_param("clf_class_weight", clf_params['class_weight'])
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("label_cols", label_cols)

    # Создаём и обучаем пайплайн
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', MultiOutputClassifier(LogisticRegression(**clf_params)))
    ])

    pipeline.fit(X_train, y_train)

    # Предсказания
    y_pred = pipeline.predict_proba(X_val)

    # Считаем AUC для каждого класса
    auc_scores = {}
    for i, col in enumerate(label_cols):
        auc = roc_auc_score(y_val[col], y_pred[i][:, 1])
        auc_scores[col] = auc
        print(f"{col}: {auc:.4f}")

    mean_auc = sum(auc_scores.values()) / len(auc_scores)
    print(f"Mean ROC-AUC: {mean_auc:.4f}")

    # Логируем метрики
    for col, auc in auc_scores.items():
        mlflow.log_metric(f"auc_{col}", auc)
    mlflow.log_metric("mean_roc_auc", mean_auc)

    # Сохраняем модель через joblib
    joblib.dump(pipeline, 'app/models/toxic_pipeline.pkl')

    # Логируем модель в MLflow
    mlflow.sklearn.log_model(pipeline, "toxic_pipeline")

    # Логируем артефакт (файл модели)
    mlflow.log_artifact('app/models/toxic_pipeline.pkl')