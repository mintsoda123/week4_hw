import numpy as np
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QLineEdit,
)
from labs.base_lab import BaseLab


def _true_fn(x):
    return np.sin(2 * x) + 0.5 * x


class Lab3(BaseLab):

    def _build_extra_params(self):
        box = QGroupBox("Lab 3 전용 — 모델 구조")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Underfit Layers:"), 0, 0)
        self.edit_underfit = QLineEdit("4")
        grid.addWidget(self.edit_underfit, 0, 1)

        grid.addWidget(QLabel("GoodFit Layers:"), 1, 0)
        self.edit_good = QLineEdit("32, 16")
        grid.addWidget(self.edit_good, 1, 1)

        grid.addWidget(QLabel("Overfit Layers:"), 2, 0)
        self.edit_overfit = QLineEdit("256, 128, 64, 32")
        grid.addWidget(self.edit_overfit, 2, 1)

        return box

    def _parse_layers(self, text):
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            return [32]

    def get_params(self):
        params = super().get_params()
        params['underfit_layers'] = self._parse_layers(self.edit_underfit.text())
        params['good_layers'] = self._parse_layers(self.edit_good.text())
        params['overfit_layers'] = self._parse_layers(self.edit_overfit.text())
        return params

    def build_model(self, params):
        """Good Fit 모델만 실시간 학습 (기준 모델)."""
        from tensorflow import keras
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for units in params['good_layers']:
            model.add(keras.layers.Dense(units, activation=params['activation']))
            model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def generate_data(self, params):
        x_train = np.random.uniform(-2, 2, 100).reshape(-1, 1)
        y_train = _true_fn(x_train) + np.random.normal(0, 0.3, x_train.shape)
        x_val = np.random.uniform(-2, 2, 50).reshape(-1, 1)
        y_val = _true_fn(x_val) + np.random.normal(0, 0.3, x_val.shape)
        x_vis = np.linspace(-2, 2, 200).reshape(-1, 1)
        return x_train, y_train, x_val, y_val, x_vis

    def update_prediction_plot(self, ax, predictions, x_vis):
        y_true = _true_fn(x_vis)
        ax.clear()
        ax.plot(x_vis, y_true, 'k-', linewidth=2, label='True', alpha=0.7)
        ax.plot(x_vis, predictions, 'g--', linewidth=2, label='Good Fit (실시간)')
        ax.set_title("과적합 데모 — Good Fit 모델 예측")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
