import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Загружаем необходимые ресурсы (выполнить один раз)
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt_tab')
#nltk.download('stopwords')

row_data = pd.read_csv('data/train.csv')
print(row_data['comment_text'][0])

stop_words = set(stopwords.words('english'))  # только английские!

def clean_text(text):
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch.isalpha() or ch.isspace())
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


row_data['comment_text'] = row_data['comment_text'].apply(clean_text)
row_data['lenght_symbols'] = row_data['comment_text'].str.len()
row_data['tokens'] = row_data['comment_text'].str.split()
row_data['lenght_words'] = row_data['tokens'].apply(len)

print(row_data['lenght_symbols'][0])
print(row_data['tokens'][0])
print(row_data['lenght_words'][0])
print(row_data['comment_text'][0])
# Удаляем строки, где comment_text == NaN
row_data = row_data.dropna(subset=['comment_text'])

row_data.to_csv('data/cleaned_data.csv')