# week4/workers/train_worker.py
import numpy as np
from PySide6.QtCore import QThread, Signal, QCoreApplication


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

    def wait(self, msecs=None):
        """Wait for the thread and then flush queued signals."""
        if msecs is not None:
            result = super().wait(msecs)
        else:
            result = super().wait()
        QCoreApplication.processEvents()
        return result

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
