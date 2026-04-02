import numpy as np
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QLineEdit, QComboBox,
)
from labs.base_lab import BaseLab


_FUNCTIONS = {
    "sin(x)": lambda x: np.sin(x),
    "cos(x)+0.5sin(2x)": lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
    "x·sin(x)": lambda x: x * np.sin(x),
    "극한 복잡도": lambda x: (
        np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.cos(3 * x)
        + 0.2 * np.sin(5 * x) + 0.1 * x * np.cos(x)
    ),
}


class Lab1(BaseLab):

    def _build_extra_params(self):
        box = QGroupBox("Lab 1 전용")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Hidden Layers:"), 0, 0)
        self.edit_layers = QLineEdit("128, 128, 64")
        grid.addWidget(self.edit_layers, 0, 1)

        grid.addWidget(QLabel("함수 선택:"), 1, 0)
        self.combo_func = QComboBox()
        for name in _FUNCTIONS:
            self.combo_func.addItem(name, name)
        grid.addWidget(self.combo_func, 1, 1)

        return box

    def get_params(self):
        params = super().get_params()
        raw = self.edit_layers.text()
        try:
            params['hidden_layers'] = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            params['hidden_layers'] = [128, 128, 64]
        params['func_name'] = self.combo_func.currentData()
        return params

    def build_model(self, params):
        from tensorflow import keras
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for units in params['hidden_layers']:
            model.add(keras.layers.Dense(
                units,
                activation=params['activation'],
            ))
            if params['dropout'] > 0:
                model.add(keras.layers.Dropout(params['dropout']))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def generate_data(self, params):
        func = _FUNCTIONS[params['func_name']]
        x_train = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
        x_vis = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
        y_train = func(x_train)
        # 검증용: x_train 섞어서 분리
        idx = np.random.permutation(len(x_train))
        split = int(len(idx) * 0.8)
        x_val = x_train[idx[split:]]
        y_val = y_train[idx[split:]]
        x_t = x_train[idx[:split]]
        y_t = y_train[idx[:split]]
        return x_t, y_t, x_val, y_val, x_vis

    def update_prediction_plot(self, ax, predictions, x_vis):
        params = self.get_params()
        func = _FUNCTIONS[params['func_name']]
        y_true = func(x_vis)

        ax.clear()
        ax.plot(x_vis, y_true, 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(x_vis, predictions, 'r--', linewidth=2, label='Predicted')
        ax.set_title(f"1D 함수 근사: {params['func_name']}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
