# TRD: Week 4 Physics Neural Network Dashboard

**버전:** 1.0  
**작성일:** 2026-04-06  
**상태:** 완료

---

## 1. 시스템 개요

### 1.1 기술 스택

| 분류 | 기술 | 버전 |
|-----|-----|-----|
| GUI 프레임워크 | PySide6 | 6.x |
| 그래프 렌더링 | matplotlib (FigureCanvasQTAgg) | 3.x |
| 머신러닝 | TensorFlow / Keras | 2.x |
| 수치 연산 | NumPy | 1.x / 2.x |
| 런타임 | Python | 3.11 |
| 패키지 관리 | uv / pip | — |

### 1.2 실행 환경

- OS: Windows 10/11 (CPU 전용; TF >= 2.11은 Windows native GPU 미지원)
- 한글 폰트: Malgun Gothic (Windows 기본 설치)
- TF 로그 억제: `TF_CPP_MIN_LOG_LEVEL=2`

---

## 2. 아키텍처

### 2.1 디렉토리 구조

```
week4/
├── app.py                    # QApplication 진입점, matplotlib 한글 폰트 설정
├── main_window.py            # MainWindow — 사이드바 + 콘텐츠 영역
├── labs/
│   ├── __init__.py
│   ├── base_lab.py           # BaseLab 추상 위젯
│   ├── lab1_perfect1d.py     # Lab 1
│   ├── lab2_projectile.py    # Lab 2
│   ├── lab3_overfitting.py   # Lab 3
│   └── lab4_pendulum.py      # Lab 4
├── workers/
│   ├── __init__.py
│   └── train_worker.py       # QThread 학습 워커
└── tests/
    ├── __init__.py
    ├── test_labs.py
    └── test_workers.py
```

### 2.2 컴포넌트 의존 관계

```
app.py
  └── MainWindow (main_window.py)
        ├── Lab1 (labs/lab1_perfect1d.py)
        ├── Lab2 (labs/lab2_projectile.py)
        ├── Lab3 (labs/lab3_overfitting.py)
        └── Lab4 (labs/lab4_pendulum.py)
              └── BaseLab (labs/base_lab.py)
                    └── TrainWorker (workers/train_worker.py)
```

---

## 3. 모듈 명세

### 3.1 `app.py`

**역할:** QApplication 초기화, matplotlib 한글 폰트 전역 설정, MainWindow 생성 및 실행

**핵심 설정:**
```python
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

### 3.2 `main_window.py` — `MainWindow`

**역할:** 사이드바 + 콘텐츠 영역 레이아웃, Lab 위젯 생성 및 전환

**주요 속성:**

| 속성 | 타입 | 설명 |
|-----|-----|-----|
| `_labs` | `dict[str, BaseLab]` | `{"lab1": Lab1(), ...}` — 앱 시작 시 전부 생성 |
| `_sidebar_btns` | `dict[str, QPushButton]` | 사이드바 토글 버튼 |

**`_switch_lab(key)`:** 해당 Lab만 `setVisible(True)`, 나머지 `False`. 버튼 `setChecked` 동기화.

**레이아웃:**
- 사이드바: `QFrame`, 고정 너비 150px, Lab 버튼 64px 높이
- 콘텐츠: `QWidget` + `QVBoxLayout`, stretch=1

---

### 3.3 `labs/base_lab.py` — `BaseLab(QWidget)`

**역할:** 모든 Lab 공통 UI 및 학습 제어 로직. 추상 클래스.

#### 레이아웃 구조

```
QHBoxLayout (root)
├── QVBoxLayout (graph_layout, stretch=3)
│   ├── FigureCanvasQTAgg (loss_canvas)   — Train/Val Loss 곡선
│   └── FigureCanvasQTAgg (pred_canvas)   — 예측 결과
└── QVBoxLayout (ctrl_layout, stretch=1, AlignTop)
    ├── QGroupBox "지표"
    │   ├── lbl_mse, lbl_mae, lbl_mape (Lab4만 visible)
    │   └── lbl_epoch
    ├── QGroupBox "하이퍼파라미터"
    │   ├── spin_epochs (QSpinBox, 100~10000, 기본 200)
    │   ├── combo_lr (QComboBox, 0.1/0.01/0.001/0.0001, 기본 0.001)
    │   ├── combo_batch (QComboBox, 16/32/64/128/256, 기본 32)
    │   ├── slider_dropout (QSlider, 0~50, 기본 10 → 0.10)
    │   └── combo_activation (QComboBox, relu/tanh/sigmoid)
    ├── [_build_extra_params() 반환값 — 서브클래스 전용 파라미터]
    └── QHBoxLayout (버튼)
        ├── btn_start "▶ 학습 시작"
        ├── btn_stop "■ 중지"
        └── btn_reset "↺ 리셋"
```

#### 추상 메서드

| 메서드 | 시그니처 | 설명 |
|-------|---------|-----|
| `build_model` | `(params: dict) -> keras.Model` | 컴파일된 모델 반환 |
| `generate_data` | `(params: dict) -> (X_train, Y_train, X_val, Y_val, X_vis)` | 학습/검증/시각화 데이터 반환 |
| `update_prediction_plot` | `(ax, predictions, x_vis)` | pred_ax에 예측 결과 그리기 |

#### 오버라이드 훅

| 메서드 | 기본값 | 설명 |
|-------|-------|-----|
| `_build_extra_params()` | `return None` | Lab 전용 파라미터 QGroupBox 반환 |

#### `get_params()` 반환 dict

```python
{
    'epochs': int,          # spin_epochs 값
    'learning_rate': float, # combo_lr currentData
    'batch_size': int,      # combo_batch currentData
    'dropout': float,       # slider_dropout / 100
    'activation': str,      # combo_activation currentData
}
```

#### 학습 플로우

```
_on_start()
  → generate_data(params) 호출 (메인 스레드)
  → TrainWorker(build_model, data_fn, params) 생성
  → worker.progress → _on_progress() 연결
  → worker.finished → _on_finished() 연결
  → worker.start()

_on_progress(epoch, train_loss, val_loss, predictions)
  → _train_losses / _val_losses 리스트에 append
  → loss_canvas 갱신 (Train + Val Loss 곡선)
  → update_prediction_plot() 호출 → pred_canvas 갱신
  → lbl_epoch 갱신

_on_finished(metrics: dict)
  → lbl_mse, lbl_mae 갱신
  → 버튼 상태 복원 (start 활성, stop 비활성)
```

---

### 3.4 `workers/train_worker.py` — `TrainWorker(QThread)`

**역할:** TensorFlow 학습을 백그라운드 스레드에서 실행하고, epoch마다 Signal을 emit해 UI를 갱신한다.

#### 생성자 파라미터

| 파라미터 | 타입 | 설명 |
|---------|-----|-----|
| `model_fn` | `callable` | `params → keras.Model` |
| `data_fn` | `callable` | `params → (X_train, Y_train, X_val, Y_val, X_vis)` |
| `params` | `dict` | 최소한 `epochs`, `batch_size` 포함 |

#### Signals

| Signal | 파라미터 | 발생 시점 |
|--------|---------|---------|
| `progress` | `(epoch: int, train_loss: float, val_loss: float, predictions: np.ndarray)` | 매 epoch 종료 |
| `finished` | `(metrics: dict)` — `{'mse': float, 'mae': float}` | 학습 완료 |

#### 중단 메커니즘

`stop()` 호출 시 `_stop_flag = True` 설정 → `_EpochCallback.on_epoch_end()`에서 `model.stop_training = True` 설정 → Keras가 다음 epoch 전에 학습 종료.

#### 최종 지표 계산

학습 완료 후 `X_val`(없으면 `X_train`)으로 예측 후 MSE/MAE 계산:
```python
mse = float(np.mean((y_pred - y_true) ** 2))
mae = float(np.mean(np.abs(y_pred - y_true)))
```

---

### 3.5 Lab 서브클래스 명세

#### Lab 1 — `lab1_perfect1d.py`

| 항목 | 내용 |
|-----|-----|
| 입력 shape | `(1,)` — x 값 |
| 출력 shape | `(1,)` — y 값 |
| 학습 데이터 | `np.linspace(-2π, 2π, 200)`, 80/20 permutation split |
| 시각화 데이터 | `np.linspace(-2π, 2π, 400)` |
| 선택 함수 | `sin(x)` / `cos(x)+0.5sin(2x)` / `x·sin(x)` / 극한 복잡도 |
| 모델 구조 | 입력 → [Dense(units, act) + Dropout] × N → Dense(1, linear) |
| 추가 파라미터 | `hidden_layers` (QLineEdit, 예: `128, 128, 64`), `func_name` (QComboBox) |

#### Lab 2 — `lab2_projectile.py`

| 항목 | 내용 |
|-----|-----|
| 입력 shape | `(3,)` — `[v₀, θ, t]` |
| 출력 shape | `(2,)` — `[x, y]` |
| 물리 공식 | `x = v₀·cos(θ)·t`, `y = v₀·sin(θ)·t - 0.5·g·t²` (g=9.81) |
| 학습 데이터 | n_samples개 (v₀∈[10,50], θ∈[20,70], t∈[0, 0.9·t_max]), y≥0 필터 |
| 고정 모델 구조 | `[128, 64, 32]` + Dropout × 3 + Dense(2, linear) |
| 추가 파라미터 | `n_samples`, `noise_level`, `test_v0`, `test_theta` |

#### Lab 3 — `lab3_overfitting.py`

| 항목 | 내용 |
|-----|-----|
| 입력 shape | `(1,)` |
| 출력 shape | `(1,)` |
| 참 함수 | `sin(2x) + 0.5x` + N(0, 0.3) 노이즈 |
| 학습 데이터 | train 100개, val 50개, `x ∈ [-2, 2]` |
| 실시간 학습 모델 | Good Fit (`good_layers` + Dropout(0.2)) |
| 추가 파라미터 | `underfit_layers`, `good_layers`, `overfit_layers` |

#### Lab 4 — `lab4_pendulum.py`

| 항목 | 내용 |
|-----|-----|
| 입력 shape | `(2,)` — `[L, θ₀]` |
| 출력 shape | `(1,)` — 주기 T (초) |
| 물리 공식 | `T = 2π√(L/g) · [1 + θ²/16 + 11θ⁴/3072]` |
| 학습 데이터 | 2000개 (L∈[0.5,3.0], θ∈[5,80]), T에 1% 가우시안 노이즈 |
| 고정 모델 구조 | `[64, 32, 16]` + Dropout × 3 + Dense(1, linear) |
| 추가 지표 | MAPE(%) = `mean(|pred - true| / true) × 100` |
| 추가 파라미터 | `L`, `theta_min`, `theta_max` |

---

## 4. 데이터 흐름

```
사용자 파라미터 입력
        │
        ▼
BaseLab._on_start()
  ├─ generate_data(params)  ←── 메인 스레드, 데이터 생성
  └─ TrainWorker.start()    ←── QThread 시작
              │
              ▼ (백그라운드 스레드)
        model_fn(params)    ←── Keras 모델 빌드
        model.fit(...)
              │  (매 epoch)
              ▼
        _EpochCallback.on_epoch_end()
          └─ progress.emit(epoch, loss, val_loss, predictions)
                    │
                    ▼ (Qt Signal → 메인 스레드)
              _on_progress()
                ├─ loss_canvas 갱신
                └─ pred_canvas 갱신
              │
              ▼ (학습 완료)
        finished.emit({'mse': ..., 'mae': ...})
                    │
                    ▼
              _on_finished()
                └─ 지표 레이블 갱신, 버튼 복원
```

---

## 5. 스레딩 모델

| 스레드 | 역할 | 주의사항 |
|-------|-----|---------|
| 메인 스레드 (Qt) | UI 렌더링, Signal 핸들러, `generate_data()` | matplotlib `draw()` 는 반드시 메인 스레드에서 |
| QThread (TrainWorker) | TensorFlow `model.fit()` | UI 위젯 직접 접근 금지 — Signal로만 통신 |

Signal/Slot 연결은 Qt의 `AutoConnection` 방식으로, `progress`/`finished` emit은 QThread에서 발생하지만 핸들러는 메인 스레드의 이벤트 루프에서 실행되어 스레드 안전성이 보장됩니다.

---

## 6. 테스트

### 6.1 단위 테스트

| 파일 | 테스트 | 검증 내용 |
|-----|-------|---------|
| `tests/test_workers.py` | `test_train_worker_emits_progress` | 3 epoch 학습 시 `progress` signal 3회 emit, `finished` 1회 emit |
| `tests/test_workers.py` | `test_train_worker_stops_early` | `stop()` 호출 시 100 epoch 전에 종료 |
| `tests/test_labs.py` | `test_base_lab_has_required_widgets` | `btn_start`, `btn_stop`, `btn_reset`, `lbl_mse`, `lbl_mae`, `loss_canvas`, `pred_canvas` 존재 |
| `tests/test_labs.py` | `test_base_lab_default_params` | `get_params()` 키 확인, `epochs=200`, `batch_size=32` 기본값 |

### 6.2 테스트 실행

```bash
cd week4
python -m pytest tests/ -v
```

예상 결과: `4 passed`

---

## 7. 알려진 제약 및 한계

| 항목 | 내용 |
|-----|-----|
| GPU 미지원 | TF >= 2.11은 Windows native에서 GPU 불가. WSL2 사용 시 가능 |
| Lab 3 단순화 | Underfit/Overfit 모델은 실시간 시각화 없음, Good Fit만 표시 |
| oneDNN 경고 | TF 내부 최적화 메시지 — 기능에 영향 없음 (`TF_ENABLE_ONEDNN_OPTS=0`으로 억제 가능) |
| 학습 결과 미저장 | 창을 닫으면 학습 결과 소멸. PNG 저장 기능 미구현 |
