from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Загрузка модели (один раз при старте)
MODEL_PATH = Path("models/toxic_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)

# Список классов (для вывода)
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


@app.get("/")
def home(request: Request):
    """Главная страница с формой"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: Request, text: str = Form(...)):
    """Обработка формы: предсказание и показ результата"""
    # Предсказание
    proba = pipeline.predict_proba([text])

    # Собираем результаты в словарь
    results = {}
    for i, label in enumerate(LABELS):
        # proba[i] — это массив [P(0), P(1)] для каждого класса
        results[label] = round(proba[i][0][1], 4)  # Берём вероятность класса 1

    # Возвращаем ту же страницу с результатом
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": text,
        "results": results
    })