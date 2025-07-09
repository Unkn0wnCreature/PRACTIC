import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from preparation import data, target
import time


X, y = data, target

# Разделение на обучающую и тестовую выборки (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Нормализация данных (kNN чувствителен к масштабу)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Список метрик для сравнения
METRICS = {
    'Евклидова': 'euclidean',
    'Манхэттенская': 'manhattan',
    'Чебышева': 'chebyshev',
    'Косинусная': 'cosine'
}

# Сравнение точности для разных метрик
results = {}
results_time = {}
for name, metric in METRICS.items():
    
    knn = KNeighborsClassifier(n_neighbors=7, metric=metric)
    start = time.time()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    total_time = time.time() - start
    results[name] = accuracy
    results_time[name] = total_time
    print(f"{name} метрика: Точность = {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["extroverts", "Introverts"]))


# График точности

plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.title(f"Сравнение точности kNN с разными метриками (Extrovert vs. Introvert Behavior Data)")
plt.ylabel('Точность')
plt.ylim(0.9, 0.94)
plt.grid(True, 'both', 'y')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.title(f"Сравнение скорости выполнения kNN с разными метриками (Extrovert vs. Introvert Behavior Data)")
plt.ylabel('Длительность')
plt.grid(True, 'both', 'y')
plt.show()