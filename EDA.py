import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import ast

data = pd.read_csv('data/cleaned_data.csv')

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 5))
sns.histplot(data['lenght_words'], bins=5, kde=True, color='royalblue')
plt.title('Гистограмма и плотность распределения')
plt.xlabel('Длина')
plt.ylabel('Частота')
#plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=data['lenght_words'], color='orange')
plt.title('Диаграмма размаха (Boxplot)')
plt.xlabel('Длина')
#plt.show()


data['tokens'] = data['tokens'].apply(ast.literal_eval)
toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
def get_top_words(tokens_series, n=20):
    words = [word for tokens in tokens_series for word in tokens]
    return Counter(words).most_common(n)
for label in toxic_types:
    subset = data[data[label] == 1]['tokens']
    top_words = get_top_words(subset)
    print(f"Топ-30 для {label}:", top_words)


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for col in label_cols:
    count = data[col].sum()
    percent = 100 * count / len(data)
    print(f"{col:15} {count:6} комментариев ({percent:.2f}%)")

