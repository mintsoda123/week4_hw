# Week 4 Physics Neural Network Dashboard — Design Spec

**Date:** 2026-04-01
**Status:** Approved

---

## Overview

A PySide6 desktop application that wraps the four Week 4 physics Neural Network labs into a unified learning dashboard. Users can adjust hyperparameters, run training in the background, and watch Loss curves and prediction results update in real time.

---

## Layout

```
┌─────────────────────────────────────────────────────────┐
│  Week 4: Physics Neural Network Dashboard               │
├──────────┬──────────────────────────────────────────────┤
│  [Lab 1] │  ┌─────────────────┬────────────────────┐   │
│  [Lab 2] │  │  Loss 곡선      │  예측 결과 그래프  │   │
│  [Lab 3] │  │  (실시간 업데이트) │                  │   │
│  [Lab 4] │  ├─────────────────┴────────────────────┤   │
│          │  │  지표: MSE / MAE / MAPE              │   │
│          │  ├──────────────────────────────────────┤   │
│          │  │  하이퍼파라미터 패널                 │   │
│          │  │  Epochs / LR / Hidden Layers /       │   │
│          │  │  Dropout / Batch Size / Activation   │   │
│          │  ├──────────────────────────────────────┤   │
│          │  │  [▶ 학습 시작]  [■ 중지]  [↺ 리셋]  │   │
└──────────┴──────────────────────────────────────────────┘
```

- Left sidebar: Lab 1 ~ Lab 4 navigation buttons
- Main area (right): per-Lab content panel

---

## Architecture

```
week4/
├── app.py                    # QApplication 진입점
├── main_window.py            # MainWindow — 사이드바 + 콘텐츠 영역
├── labs/
│   ├── base_lab.py           # 공통 Lab 위젯 (그래프, 지표, 버튼)
│   ├── lab1_perfect1d.py     # Lab 1 전용 파라미터 + 학습 로직
│   ├── lab2_projectile.py    # Lab 2 전용
│   ├── lab3_overfitting.py   # Lab 3 전용
│   └── lab4_pendulum.py      # Lab 4 전용
└── workers/
    └── train_worker.py       # QThread 기반 학습 워커
```

---

## Data Flow

1. 사용자가 하이퍼파라미터 설정 → `[학습 시작]` 클릭
2. `TrainWorker(QThread)` 생성 → 백그라운드에서 TensorFlow 학습 실행
3. 매 epoch 후 `progress_signal(epoch, loss, val_loss, predictions)` emit
4. 메인 스레드: matplotlib `FigureCanvasQTAgg` 캔버스 업데이트
5. `[중지]` 클릭 시 워커에 중단 플래그 전달 → 학습 조기 종료
6. `[리셋]` 클릭 시 그래프·지표 초기화, 파라미터 기본값 복원

---

## Graph Implementation

- `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg` 를 PySide6 위젯으로 임베딩
- 각 Lab 패널에 2개 캔버스:
  - **좌측**: Train / Validation Loss 실시간 곡선
  - **우측**: 현재 모델의 예측 결과 (epoch마다 갱신)
- 학습 완료 후 최종 그래프를 `outputs/` 디렉토리에 PNG 저장

---

## Hyperparameter Panel

### 공통 (전 Lab)

| 파라미터 | 위젯 | 범위 / 옵션 |
|---|---|---|
| Epochs | QSpinBox | 100 ~ 10000 |
| Learning Rate | QComboBox | 0.1 / 0.01 / 0.001 / 0.0001 |
| Batch Size | QComboBox | 16 / 32 / 64 / 128 / 256 |
| Dropout | QSlider | 0.0 ~ 0.5 (0.05 단위) |
| Activation | QComboBox | relu / tanh / sigmoid |

### Lab 전용

**Lab 1 — 1D 함수 근사**
- Hidden Layers: QLineEdit (예: `128, 128, 64`)
- 학습할 함수 선택: QComboBox (sin(x) / cos(x)+0.5sin(2x) / x·sin(x) / 극한 복잡도)

**Lab 2 — 포물선 운동**
- 샘플 수: QSpinBox (500 ~ 5000)
- 노이즈 레벨: QDoubleSpinBox (0.0 ~ 2.0)
- 테스트 조건: v₀ (QSpinBox), θ (QSpinBox)

**Lab 3 — 과적합 데모**
- 비교 모델 3개 각각의 Hidden Layers: QLineEdit × 3
- (Underfit / Good Fit / Overfit 레이블 표시)

**Lab 4 — 진자 주기**
- 진자 길이 L: QDoubleSpinBox (0.1 ~ 5.0m)
- 초기 각도 범위: QSpinBox min / max (5° ~ 80°)

---

## Metrics Panel

학습 완료 또는 매 N epoch마다 갱신:

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error, Lab 4 전용)
- 현재 Epoch / 전체 Epoch
- 경과 시간

---

## Control Buttons

| 버튼 | 동작 |
|---|---|
| ▶ 학습 시작 | TrainWorker 스레드 시작, 버튼 비활성화 |
| ■ 중지 | 워커에 stop 플래그 전달, 학습 조기 종료 |
| ↺ 리셋 | 그래프·지표 초기화, 파라미터 기본값 복원 |

---

## Tech Stack

- **GUI**: PySide6
- **Graphs**: matplotlib (`FigureCanvasQTAgg`)
- **ML**: TensorFlow / Keras (기존 4개 py 파일 로직 재활용)
- **Threading**: `QThread` + `Signal`

---

## Files to Create

| 파일 | 설명 |
|---|---|
| `week4/app.py` | 진입점 |
| `week4/main_window.py` | 메인 윈도우 |
| `week4/labs/base_lab.py` | 공통 Lab 위젯 |
| `week4/labs/lab1_perfect1d.py` | Lab 1 |
| `week4/labs/lab2_projectile.py` | Lab 2 |
| `week4/labs/lab3_overfitting.py` | Lab 3 |
| `week4/labs/lab4_pendulum.py` | Lab 4 |
| `week4/workers/train_worker.py` | QThread 워커 |
