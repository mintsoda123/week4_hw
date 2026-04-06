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
