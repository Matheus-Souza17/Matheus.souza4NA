import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️ Carregar o conjunto de dados Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target  # Adicionando a coluna de rótulos

# Exibir as primeiras 5 linhas do dataset
print("\n Primeiras amostras do dataset:")
print(df.head())

# 2️ Separar features (X) e target (y)
X = df.drop(columns=['target'])  # Remover a coluna alvo
y = df['target']  # Classe do vinho

# 3️ Divisão treino/teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4️ Normalizar os dados (padronização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️ Treinar o modelo KNN com k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 6️ Predição nos dados de teste
y_pred = knn.predict(X_test_scaled)

# 7️ Avaliação do modelo
acc = accuracy_score(y_test, y_pred)
print(f"\n Acurácia do modelo KNN: {acc:.4f}")

# Relatório de classificação
print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n Matriz de Confusão:")
print(conf_matrix)

#  Exibir matriz de confusão como heatmap
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - KNN")
plt.show()
