import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd
df = pd.read_csv('data/raw/full_data_2021_FORD.csv')
df.head(2)


# Baixar recursos necessários
nltk.download('punkt_tab')

nltk.download('stopwords')

# Função de pré-processamento de texto
def preprocess_text(text):
    # Remover caracteres especiais e pontuações
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Converter para minúsculas
    text = text.lower()
    
    # Tokenização
    words = word_tokenize(text)
    
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming (redução das palavras às suas raízes)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Exemplo de uso
df['processed_summary'] = df['summary'].apply(preprocess_text)
print(df[['summary', 'processed_summary']].head())
