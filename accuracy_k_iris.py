import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from preparation import data, target

iris = load_iris()
X, y = iris.data, iris.target

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
res = []
for name, metric in METRICS.items():
    result = [0] + [0]*20
    for neighbors in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result[neighbors] = accuracy

    res.append(result)

plt.figure(figsize=(10, 7))
plt.plot(res[0], 'o-r', label="Евклидова", ms=3)
plt.plot(res[1], '.-g', label="Манхэттенская", ms=3)
plt.plot(res[2], 's-k', label="Чебышева", ms=3)
plt.plot(res[3], 'v-b', label="Косинусная", ms=3)
plt.legend()
plt.title("Зависимость точности предсказания от количества соседей (Iris)")
plt.ylabel("Точность")
plt.xlim(0.5, 15)
plt.ylim(0.7, 1.05)
plt.grid(True)
plt.show()