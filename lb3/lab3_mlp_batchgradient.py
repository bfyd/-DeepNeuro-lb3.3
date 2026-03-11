# -*- coding: utf-8 -*-
"""
Использование класса MLP (из neural.py) с алгоритмом SGD.
Исходные данные: ирисы Фишера (первые 100 строк, два признака).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Импортируем реализацию MLP из файла neural.py
from neural import MLP

# Загружаем данные
df = pd.read_csv('data.csv')

# Берём первые 100 строк, 4-й столбец (целевые значения)
y = df.iloc[0:100, 4].values
# Преобразуем в числовой формат: 1 для setosa, 0 для versicolor
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1, 1)

# Используем два признака (для удобства визуализации) – столбцы 0 и 2
# ВНИМАНИЕ: больше не добавляем столбец единиц, так как MLP не требует фиктивного признака
X = df.iloc[0:100, [0, 2]].values

# Задаём параметры сети
inputSize = X.shape[1]          # 2 признака
hiddenSizes = 5                  # число нейронов скрытого слоя
outputSize = 1                    # один выход (бинарная классификация)
learning_rate = 0.01
epochs = 50                       # количество эпох обучения

# Создаём экземпляр MLP
mlp = MLP(inputSize, outputSize, learning_rate=learning_rate, hiddenSizes=hiddenSizes)

# Обучаем сеть (используется стохастический градиентный спуск – образцы обрабатываются по одному)
mlp.train(X, y, epochs=epochs)

# Вычисляем среднюю ошибку на обучающей выборке после обучения
predictions = mlp.predict(X)
print("Средняя ошибка после обучения:", np.mean(np.square(y - predictions)))

# Точность классификации (порог 0.5)
accuracy_train = np.sum((predictions > 0.5) == y) / len(y)
print("Точность на обучающей выборке:", accuracy_train)

# --------------------------------------------------------------------
# Проверка на всей выборке (первые 100 остались, теперь все 150 строк)
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == "Iris-setosa", 1, 0).reshape(-1, 1)
X_all = df.iloc[:, [0, 2]].values   # те же два признака, без единичного столбца

# Предсказание на всех данных (обучение не производится заново)
pred_all = mlp.predict(X_all)
accuracy_all = np.sum((pred_all > 0.5) == y_all) / len(y_all)
print("Точность на всей выборке:", accuracy_all)

# Ошибка (количество неверных ответов)
errors = np.sum(np.abs(y_all - (pred_all > 0.5)))
print("Число ошибок на всей выборке:", errors)