from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QPushButton, QVBoxLayout, QWidget)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WaveLabel - 波形标注工具")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化UI
        self.init_ui()
        self.current_index = 0
        self.waveforms = []  # 加载波形数据
        
    def init_ui(self):
        # 创建matplotlib图形
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        
        # 创建控件
        self.label = QLabel("波形标注")
        self.prev_btn = QPushButton("上一个")
        self.next_btn = QPushButton("下一个")
        self.save_btn = QPushButton("保存标注")
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.save_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # 连接信号
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.save_btn.clicked.connect(self.save_annotation)
        
    def show_waveform(self, index):
        """显示指定索引的波形"""
        if 0 <= index < len(self.waveforms):
            self.ax.clear()
            self.ax.plot(self.waveforms[index])
            self.canvas.draw()
            self.current_index = index

    def show_previous(self):
        self.show_waveform(self.current_index - 1)
    
    def show_next(self):
        self.show_waveform(self.current_index + 1)
    
    def save_annotation(self):
        # 保存标注逻辑
        pass

if __name__ == "__main__":
    app = QApplication([])
    window = AnnotationTool()
    window.show()
    app.exec_()