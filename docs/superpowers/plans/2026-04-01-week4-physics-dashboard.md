# Week 4 Physics Neural Network Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PySide6 대시보드 앱을 만들어 4개의 물리 Neural Network 실험을 탭 방식으로 실행하고, 하이퍼파라미터를 조절하며 실시간으로 Loss 곡선과 예측 결과를 관찰한다.

**Architecture:** `BaseLab` 위젯이 공통 UI(matplotlib 캔버스 2개, 지표 패널, 공통 하이퍼파라미터, 버튼)를 제공하고, 각 Lab 서브클래스가 모델/데이터/예측 시각화를 오버라이드한다. `TrainWorker(QThread)`가 백그라운드에서 TF 학습을 실행하고 epoch마다 Signal을 emit해 UI를 갱신한다.

**Tech Stack:** PySide6, matplotlib (FigureCanvasQTAgg), TensorFlow/Keras, numpy

---

## File Map

| 경로 | 역할 |
|---|---|
| `week4/app.py` | QApplication 진입점 |
| `week4/main_window.py` | 탭바 + 사이드바 MainWindow |
| `week4/labs/__init__.py` | 패키지 init |
| `week4/labs/base_lab.py` | 공통 Lab 위젯 (추상 기반) |
| `week4/labs/lab1_perfect1d.py` | Lab 1 — 1D 함수 근사 |
| `week4/labs/lab2_projectile.py` | Lab 2 — 포물선 운동 |
| `week4/labs/lab3_overfitting.py` | Lab 3 — 과적합 데모 |
| `week4/labs/lab4_pendulum.py` | Lab 4 — 진자 주기 |
| `week4/workers/__init__.py` | 패키지 init |
| `week4/workers/train_worker.py` | QThread 학습 워커 |
| `week4/tests/test_workers.py` | 워커 단위 테스트 |
| `week4/tests/test_labs.py` | Lab 로직 단위 테스트 |

---

## Task 1: 프로젝트 스캐폴드

**Files:**
- Create: `week4/app.py`
- Create: `week4/labs/__init__.py`
- Create: `week4/workers/__init__.py`
- Create: `week4/tests/__init__.py`

- [ ] **Step 1: 빈 패키지 파일 생성**

```python
# week4/labs/__init__.py
# week4/workers/__init__.py
# week4/tests/__init__.py
```
세 파일 모두 빈 파일로 생성한다.

- [ ] **Step 2: app.py 작성**

```python
# week4/app.py
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Week 4: Physics Neural Network Dashboard")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행 가능 여부 확인 (main_window 없어도 import 오류만 나야 함)**

```bash
cd C:/Users/my/week4
python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
```
Expected: `PySide6 OK`

- [ ] **Step 4: Commit**

```bash
cd C:/Users/my/week4
git init  # 아직 git repo가 없으면
git add app.py labs/__init__.py workers/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure for PySide6 dashboard"
```

---

## Task 2: TrainWorker (QThread 학습 워커)

**Files:**
- Create: `week4/workers/train_worker.py`
- Create: `week4/tests/test_workers.py`

- [ ] **Step 1: 테스트 작성**

```python
# week4/tests/test_workers.py
import sys
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

app = QApplication.instance() or QApplication(sys.argv)


def test_train_worker_emits_progress():
    """워커가 epoch마다 progress 신호를 emit하는지 확인"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from workers.train_worker import TrainWorker

    received = []

    def make_model(params):
        model = keras.Sequential([
            keras.layers.Input(shape=(1,)),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def make_data(params):
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        y = x * 2
        x_vis = np.linspace(0, 1, 10).reshape(-1, 1)
        return x, y, x, y, x_vis

    params = {'epochs': 3, 'batch_size': 8}
    worker = TrainWorker(make_model, make_data, params)
    worker.progress.connect(lambda ep, loss, val, preds: received.append(ep))

    done = []
    worker.finished.connect(lambda metrics: done.append(metrics))
    worker.start()
    worker.wait(10000)

    assert len(received) == 3, f"Expected 3 progress signals, got {len(received)}"
    assert len(done) == 1


def test_train_worker_stops_early():
    """stop() 호출 시 학습이 중단되는지 확인"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from workers.train_worker import TrainWorker

    received = []

    def make_model(params):
        model = keras.Sequential([
            keras.layers.Input(shape=(1,)),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def make_data(params):
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        y = x * 2
        x_vis = np.linspace(0, 1, 10).reshape(-1, 1)
        return x, y, x, y, x_vis

    params = {'epochs': 100, 'batch_size': 8}
    worker = TrainWorker(make_model, make_data, params)
    worker.progress.connect(lambda ep, loss, val, preds: received.append(ep))

    worker.start()
    import time
    time.sleep(0.3)
    worker.stop()
    worker.wait(5000)

    assert len(received) < 100, "Worker should have stopped early"
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
cd C:/Users/my/week4
python -m pytest tests/test_workers.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'workers.train_worker'`

- [ ] **Step 3: TrainWorker 구현**

```python
# week4/workers/train_worker.py
import numpy as np
from PySide6.QtCore import QThread, Signal


class TrainWorker(QThread):
    """백그라운드에서 TensorFlow 학습을 실행하는 워커.

    Signals:
        progress(epoch, train_loss, val_loss, predictions):
            매 epoch 종료 후 emit. predictions는 시각화용 numpy array.
        finished(metrics):
            학습 완료 후 emit. metrics는 {'mse', 'mae'} dict.
    """

    progress = Signal(int, float, float, object)
    finished = Signal(dict)

    def __init__(self, model_fn, data_fn, params, parent=None):
        """
        Args:
            model_fn: params -> keras.Model 을 반환하는 callable
            data_fn:  params -> (X_train, Y_train, X_val, Y_val, X_vis) 반환
            params:   dict. 최소한 'epochs', 'batch_size' 포함.
        """
        super().__init__(parent)
        self._model_fn = model_fn
        self._data_fn = data_fn
        self._params = params
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        from tensorflow import keras

        model = self._model_fn(self._params)
        X_train, Y_train, X_val, Y_val, X_vis = self._data_fn(self._params)

        worker_ref = self

        class _EpochCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if worker_ref._stop_flag:
                    self.model.stop_training = True
                    return
                logs = logs or {}
                preds = self.model.predict(X_vis, verbose=0)
                worker_ref.progress.emit(
                    epoch,
                    float(logs.get('loss', 0.0)),
                    float(logs.get('val_loss', 0.0)),
                    preds,
                )

        val_data = (X_val, Y_val) if X_val is not None else None

        model.fit(
            X_train, Y_train,
            validation_data=val_data,
            epochs=self._params['epochs'],
            batch_size=self._params['batch_size'],
            callbacks=[_EpochCallback()],
            verbose=0,
        )

        # 최종 지표 계산
        y_pred = model.predict(X_val if X_val is not None else X_train, verbose=0)
        y_true = Y_val if Y_val is not None else Y_train
        mse = float(np.mean((y_pred - y_true) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        self.finished.emit({'mse': mse, 'mae': mae})
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
cd C:/Users/my/week4
python -m pytest tests/test_workers.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add workers/train_worker.py tests/test_workers.py
git commit -m "feat: add TrainWorker QThread with per-epoch progress signals"
```

---

## Task 3: BaseLab 위젯

**Files:**
- Create: `week4/labs/base_lab.py`
- Create: `week4/tests/test_labs.py`

- [ ] **Step 1: 테스트 작성**

```python
# week4/tests/test_labs.py
import sys
import numpy as np
import pytest
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)


def test_base_lab_has_required_widgets():
    """BaseLab이 버튼, 지표 레이블, 캔버스를 가지는지 확인"""
    from labs.base_lab import BaseLab

    class DummyLab(BaseLab):
        def build_model(self, params):
            import tensorflow as tf
            from tensorflow import keras
            model = keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        def generate_data(self, params):
            x = np.linspace(0, 1, 30).reshape(-1, 1)
            y = x * 2
            return x, y, x, y, x

        def update_prediction_plot(self, ax, predictions, x_vis):
            ax.clear()
            ax.plot(x_vis, predictions)

    lab = DummyLab()
    assert lab.btn_start is not None
    assert lab.btn_stop is not None
    assert lab.btn_reset is not None
    assert lab.lbl_mse is not None
    assert lab.lbl_mae is not None
    assert lab.loss_canvas is not None
    assert lab.pred_canvas is not None


def test_base_lab_default_params():
    """기본 파라미터 값 확인"""
    from labs.base_lab import BaseLab

    class DummyLab(BaseLab):
        def build_model(self, params):
            import tensorflow as tf
            from tensorflow import keras
            model = keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        def generate_data(self, params):
            x = np.linspace(0, 1, 30).reshape(-1, 1)
            y = x * 2
            return x, y, x, y, x

        def update_prediction_plot(self, ax, predictions, x_vis):
            pass

    lab = DummyLab()
    params = lab.get_params()
    assert 'epochs' in params
    assert 'batch_size' in params
    assert 'learning_rate' in params
    assert 'dropout' in params
    assert 'activation' in params
    assert params['epochs'] == 200
    assert params['batch_size'] == 32
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
cd C:/Users/my/week4
python -m pytest tests/test_labs.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'labs.base_lab'`

- [ ] **Step 3: BaseLab 구현**

```python
# week4/labs/base_lab.py
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

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

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
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
cd C:/Users/my/week4
python -m pytest tests/test_labs.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add labs/base_lab.py tests/test_labs.py
git commit -m "feat: add BaseLab widget with matplotlib canvases and hyperparameter controls"
```

---

## Task 4: Lab 1 — 1D 함수 근사

**Files:**
- Create: `week4/labs/lab1_perfect1d.py`

- [ ] **Step 1: Lab1 구현**

```python
# week4/labs/lab1_perfect1d.py
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
        import tensorflow as tf
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
```

- [ ] **Step 2: 빠른 수동 확인**

```bash
cd C:/Users/my/week4
python -c "
from PySide6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from labs.lab1_perfect1d import Lab1
lab = Lab1()
lab.show()
print('Lab1 OK')
# 바로 종료
app.quit()
"
```
Expected: `Lab1 OK` (창이 잠깐 뜨고 꺼짐)

- [ ] **Step 3: Commit**

```bash
git add labs/lab1_perfect1d.py
git commit -m "feat: add Lab1 widget for 1D function approximation"
```

---

## Task 5: Lab 2 — 포물선 운동

**Files:**
- Create: `week4/labs/lab2_projectile.py`

- [ ] **Step 1: Lab2 구현**

```python
# week4/labs/lab2_projectile.py
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
```

- [ ] **Step 2: 빠른 수동 확인**

```bash
cd C:/Users/my/week4
python -c "
from PySide6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from labs.lab2_projectile import Lab2
lab = Lab2()
lab.show()
print('Lab2 OK')
app.quit()
"
```
Expected: `Lab2 OK`

- [ ] **Step 3: Commit**

```bash
git add labs/lab2_projectile.py
git commit -m "feat: add Lab2 widget for projectile motion regression"
```

---

## Task 6: Lab 3 — 과적합 데모

**Files:**
- Create: `week4/labs/lab3_overfitting.py`

- [ ] **Step 1: Lab3 구현**

Lab 3는 3개의 모델을 순차 학습하므로 TrainWorker를 3번 사용한다. 단순화를 위해 "Good Fit" 모델만 실시간으로 시각화하고 나머지는 학습 후 예측값을 저장한다.

```python
# week4/labs/lab3_overfitting.py
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
```

- [ ] **Step 2: 빠른 수동 확인**

```bash
cd C:/Users/my/week4
python -c "
from PySide6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from labs.lab3_overfitting import Lab3
lab = Lab3()
lab.show()
print('Lab3 OK')
app.quit()
"
```
Expected: `Lab3 OK`

- [ ] **Step 3: Commit**

```bash
git add labs/lab3_overfitting.py
git commit -m "feat: add Lab3 widget for overfitting/underfitting demo"
```

---

## Task 7: Lab 4 — 진자 주기 예측

**Files:**
- Create: `week4/labs/lab4_pendulum.py`

- [ ] **Step 1: Lab4 구현**

```python
# week4/labs/lab4_pendulum.py
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
```

- [ ] **Step 2: 빠른 수동 확인**

```bash
cd C:/Users/my/week4
python -c "
from PySide6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from labs.lab4_pendulum import Lab4
lab = Lab4()
lab.show()
print('Lab4 OK, MAPE visible:', lab.lbl_mape.isVisible())
app.quit()
"
```
Expected: `Lab4 OK, MAPE visible: True`

- [ ] **Step 3: Commit**

```bash
git add labs/lab4_pendulum.py
git commit -m "feat: add Lab4 widget for pendulum period prediction"
```

---

## Task 8: MainWindow + 최종 통합

**Files:**
- Create: `week4/main_window.py`

- [ ] **Step 1: MainWindow 구현**

```python
# week4/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QPushButton, QVBoxLayout, QLabel, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from labs.lab1_perfect1d import Lab1
from labs.lab2_projectile import Lab2
from labs.lab3_overfitting import Lab3
from labs.lab4_pendulum import Lab4


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 4: Physics Neural Network Dashboard")
        self.resize(1200, 750)

        self._labs = {}
        self._active_btn = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 사이드바 ───────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setFixedWidth(150)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 16, 8, 16)
        sidebar_layout.setSpacing(8)

        title = QLabel("Physics\nNN Labs")
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        sidebar_layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(sep)

        self._sidebar_btns = {}
        lab_labels = {
            "lab1": "Lab 1\n1D 함수 근사",
            "lab2": "Lab 2\n포물선 운동",
            "lab3": "Lab 3\n과적합 데모",
            "lab4": "Lab 4\n진자 주기",
        }
        for key, label in lab_labels.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(64)
            btn.clicked.connect(lambda checked, k=key: self._switch_lab(k))
            sidebar_layout.addWidget(btn)
            self._sidebar_btns[key] = btn

        sidebar_layout.addStretch()
        layout.addWidget(sidebar)

        # ── 콘텐츠 영역 ───────────────────────────────────────────
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._content, stretch=1)

        # Lab 위젯 미리 생성
        self._labs = {
            "lab1": Lab1(),
            "lab2": Lab2(),
            "lab3": Lab3(),
            "lab4": Lab4(),
        }
        for lab in self._labs.values():
            lab.hide()
            self._content_layout.addWidget(lab)

        # 기본으로 Lab 1 표시
        self._switch_lab("lab1")

    def _switch_lab(self, key: str):
        # 이전 Lab 숨기기
        for k, lab in self._labs.items():
            lab.setVisible(k == key)

        # 버튼 상태 갱신
        for k, btn in self._sidebar_btns.items():
            btn.setChecked(k == key)
```

- [ ] **Step 2: 전체 앱 실행 확인**

```bash
cd C:/Users/my/week4
python app.py
```
Expected: 대시보드 창이 뜨고, 사이드바에서 Lab 1~4 전환이 된다. 각 Lab의 하이퍼파라미터 패널이 보인다. 학습 시작 버튼을 눌러 Loss 곡선과 예측 그래프가 실시간 갱신되는지 확인한다.

- [ ] **Step 3: Commit**

```bash
git add main_window.py
git commit -m "feat: add MainWindow with sidebar navigation and lab switching"
```

---

## Task 9: 최종 테스트 실행

- [ ] **Step 1: 전체 테스트 스위트 실행**

```bash
cd C:/Users/my/week4
python -m pytest tests/ -v
```
Expected:
```
tests/test_labs.py::test_base_lab_has_required_widgets PASSED
tests/test_labs.py::test_base_lab_default_params PASSED
tests/test_workers.py::test_train_worker_emits_progress PASSED
tests/test_workers.py::test_train_worker_stops_early PASSED
4 passed
```

- [ ] **Step 2: 수동 E2E 확인 체크리스트**

```
□ Lab 1: sin(x) 선택 → 학습 시작 → Loss 곡선 실시간 갱신 확인
□ Lab 1: 학습 중 ■ 중지 → 즉시 멈춤 확인
□ Lab 1: ↺ 리셋 → 그래프 초기화 확인
□ Lab 2: v₀, θ 변경 후 학습 → 궤적 그래프 확인
□ Lab 3: 학습 시작 → Good Fit 예측 곡선 갱신 확인
□ Lab 4: L 변경 → 학습 후 MAPE 표시 확인
□ Lab 전환 시 기존 학습 상태 유지 확인 (각 Lab 위젯이 독립)
```

- [ ] **Step 3: 최종 Commit**

```bash
git add .
git commit -m "feat: complete Week 4 Physics Neural Network Dashboard"
```
