import numpy as np
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QDoubleSpinBox, QSpinBox,
)
from labs.base_lab import BaseLab

G = 9.81


def _true_period(L, theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    T0 = 2 * np.pi * np.sqrt(L / G)
    correction = 1 + (1/16) * theta_rad**2 + (11/3072) * theta_rad**4
    return T0 * correction


class Lab4(BaseLab):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.show_mape(True)

    def _build_extra_params(self):
        box = QGroupBox("Lab 4 전용")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("진자 길이 L (m):"), 0, 0)
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.1, 5.0)
        self.spin_L.setValue(1.0)
        self.spin_L.setSingleStep(0.1)
        grid.addWidget(self.spin_L, 0, 1)

        grid.addWidget(QLabel("각도 최솟값 (°):"), 1, 0)
        self.spin_theta_min = QSpinBox()
        self.spin_theta_min.setRange(1, 40)
        self.spin_theta_min.setValue(5)
        grid.addWidget(self.spin_theta_min, 1, 1)

        grid.addWidget(QLabel("각도 최댓값 (°):"), 2, 0)
        self.spin_theta_max = QSpinBox()
        self.spin_theta_max.setRange(41, 80)
        self.spin_theta_max.setValue(80)
        grid.addWidget(self.spin_theta_max, 2, 1)

        return box

    def get_params(self):
        params = super().get_params()
        params['L'] = self.spin_L.value()
        params['theta_min'] = self.spin_theta_min.value()
        params['theta_max'] = self.spin_theta_max.value()
        params['hidden_layers'] = [64, 32, 16]
        return params

    def build_model(self, params):
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(params['hidden_layers'][0], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(params['hidden_layers'][1], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(params['hidden_layers'][2], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def generate_data(self, params):
        n = 2000
        Ls = np.random.uniform(0.5, 3.0, n)
        thetas = np.random.uniform(5, 80, n)
        T = np.array([_true_period(l, t) for l, t in zip(Ls, thetas)])
        T_noisy = T * (1 + np.random.normal(0, 0.01, n))

        X = np.column_stack([Ls, thetas])
        Y = T_noisy.reshape(-1, 1)

        idx = np.random.permutation(n)
        split = int(n * 0.8)
        X_train, Y_train = X[idx[:split]], Y[idx[:split]]
        X_val, Y_val = X[idx[split:]], Y[idx[split:]]

        # 시각화: 지정된 L에서 각도별 주기
        L_vis = params['L']
        angles_vis = np.linspace(params['theta_min'], params['theta_max'], 50)
        X_vis = np.column_stack([np.full(50, L_vis), angles_vis])

        return X_train, Y_train, X_val, Y_val, X_vis

    def update_prediction_plot(self, ax, predictions, x_vis):
        params = self.get_params()
        angles = x_vis[:, 1]
        T_true = np.array([_true_period(params['L'], a) for a in angles])

        ax.clear()
        ax.plot(angles, T_true, 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(angles, predictions.flatten(), 'r--', linewidth=2, label='NN')
        ax.set_title(f"진자 주기 예측 (L={params['L']:.1f} m)")
        ax.set_xlabel("초기 각도 (°)")
        ax.set_ylabel("주기 T (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # MAPE 계산 후 표시
        mape = np.mean(np.abs((predictions.flatten() - T_true) / T_true)) * 100
        self.lbl_mape.setText(f"{mape:.2f}%")
