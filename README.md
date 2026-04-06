# Week 4 Homework - Physics Neural Network Dashboard 🧠
202312143 이민영 week4에 대한 과제 제출입니다.

---

물리 데이터를 TensorFlow/Keras로 학습하는 4개의 Lab 실습과, 이를 하나의 PySide6 GUI 대시보드로 통합한 인터랙티브 학습 도구입니다.

---

## 📁 파일 구성 
모든 파일들은 week4 폴더에 있습니다.

```
week4/
├── app.py                    # 메인 앱 진입점 (PySide6 대시보드)
├── main_window.py            # 메인 윈도우 (사이드바 + 탭 컨테이너)
├── labs/                     # Lab 탭 위젯 모음
│   ├── base_lab.py           # 공통 Lab 위젯 (그래프, 지표, 버튼)
│   ├── lab1_perfect1d.py     # Lab 1 탭: 1D 함수 근사
│   ├── lab2_projectile.py    # Lab 2 탭: 포물선 운동
│   ├── lab3_overfitting.py   # Lab 3 탭: 과적합 데모
│   └── lab4_pendulum.py      # Lab 4 탭: 진자 주기 예측
├── workers/
│   └── train_worker.py       # QThread 기반 학습 워커
├── 01perfect1d.py            # Lab 1 단독 실행 스크립트
├── 02projectile.py           # Lab 2 단독 실행 스크립트
├── 03overfitting.py          # Lab 3 단독 실행 스크립트
├── 04pendulum.py             # Lab 4 단독 실행 스크립트
├── ex01_tanh_cubic.py        # 심화 1: 다양한 비선형 함수 추가
├── ex02_air_resistance.py    # 심화 2: 공기 저항 포함 포물선 모델
├── ex03_regularization.py    # 심화 3: L1 / L2 Regularization 구현
├── ex04_damped_pendulum.py   # 심화 4: 감쇠 진자 구현
├── docs/
│   ├── PRD.md                # 제품 요구사항 문서
│   └── TRD.md                # 기술 요구사항 문서
└── README.md                 # 이 파일
```

---

## ⚙️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install tensorflow numpy matplotlib PySide6
```

### 2. GUI 대시보드 실행
```bash
cd week4
python app.py
```

### 3. Lab 단독 실행
```bash
python 01perfect1d.py
python 02projectile.py
python 03overfitting.py
python 04pendulum.py
```

---

## 🔬 Lab 구성

| 탭 | 내용 | 핵심 기능 |
|----|------|-----------|
| Lab 1 | 1D 함수 근사 | Hidden Layer 자유 설정 + 실시간 예측 곡선 |
| Lab 2 | 포물선 운동 회귀 | v₀, θ 조절 + 궤적 시각화 |
| Lab 3 | 과적합 vs 과소적합 | 3가지 모델 구조 비교 + Loss 곡선 |
| Lab 4 | 진자 주기 예측 | RK4 시뮬레이션 + MAPE 실시간 표시 |

---

## 🔬 Lab 상세

### Lab 1 — 1D 함수 근사 (`01perfect1d.py`)

Universal Approximation Theorem을 실험합니다. 다양한 1D 함수를 Neural Network로 근사하고 네트워크 크기에 따른 성능 차이를 비교합니다.

- **함수:** `sin(x)`, `cos(x)+0.5sin(2x)`, `x·sin(x)`, 극한 복잡도 함수
- **네트워크 크기 비교:** Small [32] / Medium [64,64] / Large [128,128] / Very Large [128,128,64]
- **주의사항:** `validation_split` 사용 시 데이터 셔플링 필수 — 순차 정렬 데이터는 끝부분이 학습에서 제외됨

**생성 파일:** `outputs/perfect_1d_approximation.png`, `outputs/network_size_comparison.png`, `outputs/extreme_function_test.png`

---

### Lab 2 — 포물선 운동 회귀 (`02projectile.py`)

물리 법칙을 명시하지 않고 데이터만으로 포물선 궤적을 학습합니다.

```
x(t) = v₀·cos(θ)·t
y(t) = v₀·sin(θ)·t - 0.5·g·t²
```

- **입력:** (초기속력, 발사각도, 시간)  **출력:** (x, y 좌표)
- **테스트 조건:** v₀=20m/s θ=30° / v₀=30m/s θ=45° / v₀=40m/s θ=60°

**생성 파일:** `outputs/02_projectile_trajectories.png`, `outputs/02_projectile_analysis.png`

---

### Lab 3 — 과적합 vs 과소적합 데모 (`03overfitting.py`)

동일 함수(`y = sin(2x) + 0.5x + noise`)에 대해 3가지 모델을 비교합니다.

| 모델 | 구조 | 현상 |
|------|------|------|
| Underfit | [4] | 패턴 학습 실패, 높은 train/val loss |
| Good Fit | [32, 16] + Dropout | 최고의 일반화 성능 |
| Overfit  | [256, 128, 64, 32] | 낮은 train loss, 높은 val loss |

**생성 파일:** `outputs/03_overfitting_comparison.png`, `outputs/03_training_curves.png`, `outputs/03_error_analysis.png`

---

### Lab 4 — 진자 주기 예측 (`04pendulum.py`)

비선형 진자 운동 방정식을 Neural Network로 학습합니다.

```
작은 각도: T = 2π√(L/g)
큰 각도:   T ≈ T₀[1 + (1/16)θ₀² + ...]
운동 방정식: d²θ/dt² = -(g/L)sin(θ)
```

- **입력:** (진자 길이 L, 초기 각도 θ₀)  **출력:** 주기 T
- **목표 성능:** MAPE < 1%

**생성 파일:** `outputs/04_pendulum_prediction.png`, `outputs/04_pendulum_simulation.png`, `outputs/04_pendulum_analysis.png`

---

## 🖥️ GUI 대시보드 (`app.py`)

4개 Lab을 하나의 PySide6 창으로 통합한 실시간 학습 대시보드입니다.

- **백그라운드 학습:** QThread 기반으로 학습 중에도 UI가 멈추지 않음
- **실시간 갱신:** 매 epoch마다 Loss 곡선과 예측 그래프 업데이트
- **하이퍼파라미터 조절:** Epochs, Learning Rate, Batch Size, Dropout, Activation
- **버튼:** ▶ 학습 시작 / ■ 중지 / ↺ 리셋

---

## ⚙️ 기술 스택

- **Python** 3.11+ — 전체 구현
- **TensorFlow/Keras** — 신경망 모델 학습
- **NumPy** — 데이터 생성, 수치 계산 (RK4 적분)
- **Matplotlib** — 그래프 시각화, FigureCanvasQTAgg로 PySide6에 임베딩
- **PySide6** — GUI 대시보드

---

## 🔑 핵심 알고리즘

### Dense Layer Forward Pass
```
z = W·x + b
a = activation(z)
```

### Adam Optimizer
```
m_t = β₁·m_(t-1) + (1-β₁)·∇L
v_t = β₂·v_(t-1) + (1-β₂)·∇L²
W ← W - lr · m̂_t / (√v̂_t + ε)
```

### RK4 수치 적분 (Lab 4)
```
d²θ/dt² = -(g/L)sin(θ)
k1 = f(t, y)
k2 = f(t + h/2, y + h/2·k1)
k3 = f(t + h/2, y + h/2·k2)
k4 = f(t + h, y + h·k3)
y_next = y + h/6·(k1 + 2k2 + 2k3 + k4)
```

---

## PRD / TRD

상세 요구사항과 기술 설계는 `docs/PRD.md`, `docs/TRD.md` 파일을 참고하십시오.

---

# Week 4 스스로 해보기 — 심화 과제

> Week 4 `week4.md`의 **심화 과제** 1~4번을 직접 구현한 결과물입니다.

---

## 심화 개요

| 번호 | 파일 | 주제 | 기반 파일 |
|:----:|------|------|-----------|
| 1 | `ex01_tanh_cubic.py` | tanh(x), x³ 등 다른 함수 추가 | `01perfect1d.py` |
| 2 | `ex02_air_resistance.py` | 공기 저항 추가 모델 | `02projectile.py` |
| 3 | `ex03_regularization.py` | L1/L2 Regularization 구현 | `03overfitting.py` |
| 4 | `ex04_damped_pendulum.py` | 감쇠 진자 (Damped Pendulum) 구현 | `04pendulum.py` |

---

## 심화 1 — 다양한 비선형 함수 추가

**파일:** `ex01_tanh_cubic.py`

Lab 1의 함수 집합에 `tanh(x)`, `x³`, `x²·sin(x)`, `|x|·cos(x)` 4가지 함수를 추가하고 NN 근사 성능을 비교합니다.

| 함수 | 특성 | 근사 난이도 |
|------|------|-----------|
| `tanh(x)` | 단조 증가, 포화 구간 존재 | 쉬움 |
| `x³` | 단항식, 전역 단조 증가 | 쉬움 |
| `x²·sin(x)` | 진폭이 커지는 진동 | 중간 |
| `\|x\|·cos(x)` | x=0에서 미분 불연속 | 어려움 |

- 구조: [128, 128, 64], tanh 활성화, 3000 epochs
- **핵심 학습:** 미분 불연속점이 있는 함수(`|x|`)는 MSE가 상대적으로 크게 나타남. tanh 활성화는 부드러운 함수 근사에 유리함

**결과물:** `outputs/ex01_tanh_cubic.png`

---

## 심화 2 — 공기 저항 추가 모델

**파일:** `ex02_air_resistance.py`

선형 항력(F = -k·v)이 있는 포물선 운동을 학습하고, 이상 궤도(저항 없음)와 비교합니다.

```
공기 저항 운동 방정식:
  dvx/dt = -k·vx
  dvy/dt = -g - k·vy

해석해:
  x(t) = (vx0/k)(1 - e^(-k·t))
  y(t) = (vy0/k + g/k²)(1 - e^(-k·t)) - (g/k)·t
```

- 항력 계수: k = 0.1 s⁻¹
- 입력: (v₀, θ, t),  출력: (x, y)
- **핵심 학습:** 공기 저항 아래 최대 사거리는 45° 미만에서 발생. NN은 물리 공식 없이도 지수 감쇠가 포함된 비선형 해를 학습 가능

**결과물:** `outputs/ex02_air_resistance.png`

---

## 심화 3 — L1/L2 Regularization 구현

**파일:** `ex03_regularization.py`

소량의 학습 데이터(80개)와 큰 네트워크([128, 64, 32])를 사용해 과적합을 유도하고, 규제화 기법별 효과를 비교합니다.

| 모델 | 규제 방법 | 효과 |
|------|---------|------|
| No Reg | 없음 | 과적합 — val loss 높음 |
| L1 Reg (Lasso) | `Σ\|w\|` 페널티 | 가중치 희소화 → 불필요 뉴런 제거 |
| L2 Reg (Ridge) | `Σw²` 페널티 | 가중치 분산 억제 → 부드러운 곡선 |
| L1+L2 (Elastic Net) | 두 효과 결합 | 안정적 일반화 성능 |

```python
keras.regularizers.L1(0.01)
keras.regularizers.L2(0.01)
keras.regularizers.L1L2(l1=0.01, l2=0.01)
```

- **핵심 학습:** L2는 예측 곡선이 부드럽고 안정적. L1은 가중치 sparsity를 유도해 과적합 억제에 효과적. Dropout과 달리 규제화는 가중치 자체에 패널티를 부여함

**결과물:** `outputs/ex03_regularization.png`

---

## 심화 4 — 감쇠 진자 (Damped Pendulum)

**파일:** `ex04_damped_pendulum.py`

마찰이 있는 진자의 운동 방정식을 RK4로 수치 적분하여 학습 데이터를 생성하고, NN으로 임의의 시간 t에서의 각도를 예측합니다.

```
운동 방정식:
  d²θ/dt² = -(g/L)sin(θ) - b·(dθ/dt)
  b: 감쇠 계수
```

- 입력: (L, b, θ₀, t),  출력: θ(t)
- 4000개 샘플, RK4 dt=0.01s 적분

| 감쇠 유형 | 조건 | 물리적 특성 |
|----------|------|-----------|
| 비감쇠 | b = 0 | 진동 영속 |
| 과소감쇠 | 0 < b < 2√(g/L) | 진동하며 수렴 |
| 임계감쇠 | b = 2√(g/L) | 가장 빠른 수렴, 진동 없음 |
| 과대감쇠 | b > 2√(g/L) | 느린 수렴, 진동 없음 |

- **핵심 학습:** 위상 공간(각도 vs 각속도)에서 비감쇠는 폐곡선, 과소감쇠는 나선형으로 원점 수렴. NN은 시간에 따른 지수적 감쇠와 비선형 진동을 동시에 학습 가능

**결과물:** `outputs/ex04_damped_pendulum.png`

---

## 실행 방법

```bash
# 추가 설치 없이 실행 가능 (tensorflow, numpy, matplotlib 필요)

python ex01_tanh_cubic.py        # 약 3-5분 (3000 epochs × 4 함수)
python ex02_air_resistance.py    # 약 2-3분
python ex03_regularization.py    # 약 2-3분
python ex04_damped_pendulum.py   # 약 3-5분 (RK4 적분 포함)
```

결과 이미지는 `outputs/` 폴더에 자동 저장됩니다.

---

## 사용한 라이브러리

- `tensorflow` — Dense, Dropout, Adam, L1/L2 규제화
- `numpy` — 데이터 생성, RK4 수치 적분
- `matplotlib` — 학습 곡선, 궤적, 위상 공간 시각화
- `PySide6` — GUI 대시보드 (대시보드 실행 시 필요)
