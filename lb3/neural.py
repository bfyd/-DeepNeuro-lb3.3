import numpy as np

class MLP:
    """
    Многослойный перцептрон с одним скрытым слоем.
    Реализует обучение по алгоритму стохастического градиентного спуска.
    """
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes=5):
        """
        Параметры:
            inputSize: число входных признаков
            outputSize: размер выхода (число классов)
            learning_rate: скорость обучения
            hiddenSizes: число нейронов скрытого слоя
        """
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # вход -> скрытый
            np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # скрытый -> выход
        ]
        self.learning_rate = learning_rate
        self.layers = None

    def sigmoid(self, x):
        """Сигмоидальная функция активации."""
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        """Производная сигмоиды."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, x):
        """
        Прямой проход для одного образца x (одномерный массив).
        Сохраняет выходы всех слоёв в self.layers.
        Возвращает выход сети.
        """
        input_ = x
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0]))
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))
        self.layers = [input_, hidden_, output_]
        return output_

    def backward(self, target):
        """
        Обратный проход (backpropagation) для одного образца.
        Обновляет веса, используя сохранённые значения слоёв.
        """
        err = target - self.layers[-1]  # ошибка на выходе

        # Идём от последнего слоя к первому (исключая входной)
        for i in range(len(self.layers) - 1, 0, -1):
            err_delta = err * self.derivative_sigmoid(self.layers[i])
            err = np.dot(err_delta, self.weights[i - 1].T)
            dw = np.outer(self.layers[i - 1], err_delta)
            self.weights[i - 1] += self.learning_rate * dw

    def train(self, X, y, epochs=10, shuffle=True):
        """
        Обучение сети на наборе данных с использованием SGD.

        Параметры:
            X: матрица признаков, форма (n_samples, inputSize)
            y: вектор целевых значений, форма (n_samples,) или (n_samples, outputSize)
            epochs: количество эпох
            shuffle: перемешивать ли образцы перед каждой эпохой
        """
        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(len(X))
            else:
                indices = np.arange(len(X))

            for i in indices:
                x_i = X[i]
                target_i = y[i]
                self.feed_forward(x_i)
                self.backward(target_i)

        return self

    def predict(self, x_values):
        """Возвращает выход сети для одного или нескольких образцов."""
        # Для одного образца
        if x_values.ndim == 1:
            return self.feed_forward(x_values)
        # Для нескольких образцов (матрица)
        outputs = []
        for x in x_values:
            outputs.append(self.feed_forward(x))
        return np.array(outputs)