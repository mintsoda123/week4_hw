from abc import abstractmethod

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QSlider, QGroupBox, QSizePolicy,
)


class BaseLab(QWidget):
    """모든 Lab 위젯의 공통 기반.

    서브클래스가 반드시 오버라이드해야 하는 메서드:
        build_model(params) -> keras.Model
        generate_data(params) -> (X_train, Y_train, X_val, Y_val, X_vis)
        update_prediction_plot(ax, predictions, x_vis)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._train_losses = []
        self._val_losses = []
        self._x_vis = None

        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── 왼쪽: 그래프 영역 ─────────────────────────────────────────
        graph_layout = QVBoxLayout()

        # Loss 캔버스
        self.loss_fig = Figure(figsize=(5, 3), tight_layout=True)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_title("Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_canvas = FigureCanvasQTAgg(self.loss_fig)
        self.loss_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Prediction 캔버스
        self.pred_fig = Figure(figsize=(5, 3), tight_layout=True)
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_ax.set_title("Prediction")
        self.pred_canvas = FigureCanvasQTAgg(self.pred_fig)
        self.pred_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graph_layout.addWidget(self.loss_canvas)
        graph_layout.addWidget(self.pred_canvas)
        root.addLayout(graph_layout, stretch=3)

        # ── 오른쪽: 컨트롤 영역 ───────────────────────────────────────
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setAlignment(Qt.AlignTop)

        # 지표 패널
        metrics_box = QGroupBox("지표")
        metrics_grid = QGridLayout(metrics_box)
        metrics_grid.addWidget(QLabel("MSE:"), 0, 0)
        self.lbl_mse = QLabel("—")
        metrics_grid.addWidget(self.lbl_mse, 0, 1)
        metrics_grid.addWidget(QLabel("MAE:"), 1, 0)
        self.lbl_mae = QLabel("—")
        metrics_grid.addWidget(self.lbl_mae, 1, 1)
        self.lbl_mape_row = QLabel("MAPE:")
        self.lbl_mape = QLabel("—")
        metrics_grid.addWidget(self.lbl_mape_row, 2, 0)
        metrics_grid.addWidget(self.lbl_mape, 2, 1)
        self.lbl_mape_row.setVisible(False)
        self.lbl_mape.setVisible(False)
        self.lbl_epoch = QLabel("Epoch: 0 / 0")
        metrics_grid.addWidget(self.lbl_epoch, 3, 0, 1, 2)
        ctrl_layout.addWidget(metrics_box)

        # 공통 하이퍼파라미터
        common_box = QGroupBox("하이퍼파라미터")
        common_grid = QGridLayout(common_box)

        common_grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 10000)
        self.spin_epochs.setValue(200)
        self.spin_epochs.setSingleStep(100)
        common_grid.addWidget(self.spin_epochs, 0, 1)

        common_grid.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.combo_lr = QComboBox()
        for lr in ["0.1", "0.01", "0.001", "0.0001"]:
            self.combo_lr.addItem(lr, float(lr))
        self.combo_lr.setCurrentIndex(2)  # 0.001
        common_grid.addWidget(self.combo_lr, 1, 1)

        common_grid.addWidget(QLabel("Batch Size:"), 2, 0)
        self.combo_batch = QComboBox()
        for bs in [16, 32, 64, 128, 256]:
            self.combo_batch.addItem(str(bs), bs)
        self.combo_batch.setCurrentIndex(1)  # 32
        common_grid.addWidget(self.combo_batch, 2, 1)

        common_grid.addWidget(QLabel("Dropout:"), 3, 0)
        self.slider_dropout = QSlider(Qt.Horizontal)
        self.slider_dropout.setRange(0, 50)
        self.slider_dropout.setValue(10)
        self.slider_dropout.setTickInterval(5)
        self.lbl_dropout_val = QLabel("0.10")
        self.slider_dropout.valueChanged.connect(
            lambda v: self.lbl_dropout_val.setText(f"{v/100:.2f}")
        )
        dropout_row = QHBoxLayout()
        dropout_row.addWidget(self.slider_dropout)
        dropout_row.addWidget(self.lbl_dropout_val)
        dropout_widget = QWidget()
        dropout_widget.setLayout(dropout_row)
        common_grid.addWidget(dropout_widget, 3, 1)

        common_grid.addWidget(QLabel("Activation:"), 4, 0)
        self.combo_activation = QComboBox()
        for act in ["relu", "tanh", "sigmoid"]:
            self.combo_activation.addItem(act, act)
        common_grid.addWidget(self.combo_activation, 4, 1)

        ctrl_layout.addWidget(common_box)

        # Lab 전용 파라미터 (서브클래스가 추가)
        extra = self._build_extra_params()
        if extra is not None:
            ctrl_layout.addWidget(extra)

        # 버튼
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 학습 시작")
        self.btn_stop = QPushButton("■ 중지")
        self.btn_reset = QPushButton("↺ 리셋")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_reset)
        ctrl_layout.addLayout(btn_layout)

        root.addLayout(ctrl_layout, stretch=1)

    # ------------------------------------------------------------------ #
    # Override hooks                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_model(self, params: dict):
        """params를 받아 컴파일된 keras.Model을 반환한다."""

    @abstractmethod
    def generate_data(self, params: dict):
        """(X_train, Y_train, X_val, Y_val, X_vis) 튜플을 반환한다.
        X_vis는 예측 시각화에 사용하는 입력 배열이다.
        X_val, Y_val이 없으면 None을 반환해도 된다.
        """

    @abstractmethod
    def update_prediction_plot(self, ax, predictions, x_vis):
        """pred_ax에 현재 예측 결과를 그린다."""

    def _build_extra_params(self):
        """Lab 전용 파라미터 QGroupBox를 반환한다. 없으면 None."""
        return None

    # ------------------------------------------------------------------ #
    # Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_params(self) -> dict:
        """현재 UI 위젯 값으로 구성한 파라미터 dict를 반환한다."""
        return {
            'epochs': self.spin_epochs.value(),
            'learning_rate': self.combo_lr.currentData(),
            'batch_size': self.combo_batch.currentData(),
            'dropout': self.slider_dropout.value() / 100.0,
            'activation': self.combo_activation.currentData(),
        }

    def show_mape(self, visible: bool):
        """MAPE 행 표시 여부 제어 (Lab 4에서 사용)."""
        self.lbl_mape_row.setVisible(visible)
        self.lbl_mape.setVisible(visible)

    # ------------------------------------------------------------------ #
    # Button handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_start(self):
        from workers.train_worker import TrainWorker

        params = self.get_params()
        self._train_losses.clear()
        self._val_losses.clear()

        X_train, Y_train, X_val, Y_val, X_vis = self.generate_data(params)
        self._x_vis = X_vis

        def data_fn(_params):
            return X_train, Y_train, X_val, Y_val, X_vis

        self._worker = TrainWorker(self.build_model, data_fn, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_epoch.setText(f"Epoch: 0 / {params['epochs']}")
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_reset(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()

        self._train_losses.clear()
        self._val_losses.clear()

        self.loss_ax.clear()
        self.loss_ax.set_title("Loss")
        self.loss_canvas.draw()

        self.pred_ax.clear()
        self.pred_ax.set_title("Prediction")
        self.pred_canvas.draw()

        self.lbl_mse.setText("—")
        self.lbl_mae.setText("—")
        self.lbl_mape.setText("—")
        self.lbl_epoch.setText("Epoch: 0 / 0")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Signal handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_progress(self, epoch: int, train_loss: float, val_loss: float, predictions):
        total = self.spin_epochs.value()
        self.lbl_epoch.setText(f"Epoch: {epoch + 1} / {total}")

        self._train_losses.append(train_loss)
        if val_loss > 0:
            self._val_losses.append(val_loss)

        # Loss 캔버스 갱신
        self.loss_ax.clear()
        self.loss_ax.plot(self._train_losses, label="Train", color="blue")
        if self._val_losses:
            self.loss_ax.plot(self._val_losses, label="Val", color="red", linestyle="--")
        self.loss_ax.set_title("Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.legend(fontsize=8)
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_canvas.draw()

        # 예측 캔버스 갱신
        self.update_prediction_plot(self.pred_ax, predictions, self._x_vis)
        self.pred_canvas.draw()

        # 지표 갱신
        self.lbl_mse.setText(f"{train_loss:.6f}")

    def _on_finished(self, metrics: dict):
        self.lbl_mse.setText(f"{metrics['mse']:.6f}")
        self.lbl_mae.setText(f"{metrics['mae']:.6f}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
