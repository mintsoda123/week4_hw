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
