import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# Carrega o dataset (se necessário ajuste o encoding)
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocessamento dos dados
data.drop_duplicates(subset='v2', inplace=True)
columns_to_drop = data.columns[data.isnull().any()].tolist()
data = data.drop(columns=columns_to_drop)

data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
x = data['v2']
y = data['v1']

# Divide os dados
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vetorização do texto
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)

# Treina os modelos
logModel = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000).fit(X_train_vec, y_train)
RFModel = RandomForestClassifier(n_estimators=1000, random_state=42).fit(X_train_vec, y_train)

# Avaliação dos modelos usando funções do scikit-learn
models = [('Logistic Regression', logModel), ('Random Forest', RFModel)]

for name, model in models:
	y_pred = model.predict(X_test_vec)
	prec = precision_score(y_test, y_pred, pos_label=1)
	rec = recall_score(y_test, y_pred, pos_label=1)
	acc = accuracy_score(y_test, y_pred)
	print(f"\nModel: {name}")
	print(f"Precision: {prec:.4f}")
	print(f"Recall:    {rec:.4f}")
	print(f"Accuracy:  {acc:.4f}")
	print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

