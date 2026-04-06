import numpy as np
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox,
)
from labs.base_lab import BaseLab

G = 9.81


class Lab2(BaseLab):

    def _build_extra_params(self):
        box = QGroupBox("Lab 2 전용")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("샘플 수:"), 0, 0)
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(500, 5000)
        self.spin_samples.setValue(2000)
        self.spin_samples.setSingleStep(100)
        grid.addWidget(self.spin_samples, 0, 1)

        grid.addWidget(QLabel("노이즈 레벨:"), 1, 0)
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0.0, 2.0)
        self.spin_noise.setValue(0.5)
        self.spin_noise.setSingleStep(0.1)
        grid.addWidget(self.spin_noise, 1, 1)

        grid.addWidget(QLabel("테스트 v₀ (m/s):"), 2, 0)
        self.spin_v0 = QSpinBox()
        self.spin_v0.setRange(10, 50)
        self.spin_v0.setValue(30)
        grid.addWidget(self.spin_v0, 2, 1)

        grid.addWidget(QLabel("테스트 θ (°):"), 3, 0)
        self.spin_theta = QSpinBox()
        self.spin_theta.setRange(20, 70)
        self.spin_theta.setValue(45)
        grid.addWidget(self.spin_theta, 3, 1)

        return box

    def get_params(self):
        params = super().get_params()
        params['n_samples'] = self.spin_samples.value()
        params['noise_level'] = self.spin_noise.value()
        params['test_v0'] = self.spin_v0.value()
        params['test_theta'] = self.spin_theta.value()
        params['hidden_layers'] = [128, 64, 32]
        return params

    def build_model(self, params):
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(params['hidden_layers'][0], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(params['hidden_layers'][1], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(params['hidden_layers'][2], activation=params['activation']),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(2, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def generate_data(self, params):
        n = params['n_samples']
        noise = params['noise_level']
        v0 = params['test_v0']
        theta_deg = params['test_theta']

        # 학습 데이터
        v0s = np.random.uniform(10, 50, n)
        thetas = np.random.uniform(20, 70, n)
        theta_rads = np.deg2rad(thetas)
        t_max = 2 * v0s * np.sin(theta_rads) / G
        ts = np.random.uniform(0, t_max * 0.9, n)
        xs = v0s * np.cos(theta_rads) * ts + np.random.normal(0, noise, n)
        ys = v0s * np.sin(theta_rads) * ts - 0.5 * G * ts**2 + np.random.normal(0, noise, n)
        valid = ys >= 0
        X = np.column_stack([v0s[valid], thetas[valid], ts[valid]])
        Y = np.column_stack([xs[valid], ys[valid]])

        split = int(len(X) * 0.8)
        idx = np.random.permutation(len(X))
        X_train, Y_train = X[idx[:split]], Y[idx[:split]]
        X_val, Y_val = X[idx[split:]], Y[idx[split:]]

        # 시각화용: 테스트 조건 궤적
        theta_rad = np.deg2rad(theta_deg)
        t_vis_max = 2 * v0 * np.sin(theta_rad) / G
        t_vis = np.linspace(0, t_vis_max * 0.98, 50)
        X_vis = np.column_stack([
            np.full(50, v0),
            np.full(50, theta_deg),
            t_vis,
        ])

        return X_train, Y_train, X_val, Y_val, X_vis

    def update_prediction_plot(self, ax, predictions, x_vis):
        params = self.get_params()
        v0 = params['test_v0']
        theta_deg = params['test_theta']
        theta_rad = np.deg2rad(theta_deg)
        t_vis = x_vis[:, 2]

        x_true = v0 * np.cos(theta_rad) * t_vis
        y_true = v0 * np.sin(theta_rad) * t_vis - 0.5 * G * t_vis**2

        ax.clear()
        ax.plot(x_true, y_true, 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(predictions[:, 0], predictions[:, 1], 'r--', linewidth=2, label='NN')
        ax.set_title(f"포물선 궤적 (v₀={v0} m/s, θ={theta_deg}°)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
