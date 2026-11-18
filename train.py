import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar os dados
data = pd.read_csv("data/spam.csv")

X = data["message"]
y = data["label"]

# 2. Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Criar o modelo
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# 4. Treinar a IA
model.fit(X_train, y_train)

# 5. Avaliar o modelo
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 6. Guardar o modelo
joblib.dump(model, "spam_model.pkl")

print("\nModelo guardado com sucesso!")