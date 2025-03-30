# WaveLabel - 音频波形标注与训练系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于提取音频波形、人工标注以及训练分类模型的完整工具链。

## 功能特性

- 🎧 音频波形提取与可视化
- 🏷️ 直观的波形标注界面
- 📊 标注数据管理与导出
- 🤖 自动训练波形分类模型
- 📈 模型性能评估与可视化

## 项目结构

```bash
AudioAnnotationTool/
├── README.md               # 项目说明文档
├── requirements.txt        # Python依赖
├── config.yaml             # 配置文件
│
├── data/                   # 数据目录
│   ├── raw_audio/          # 原始音频文件
│   ├── extracted_waves/    # 提取的波形片段
│   ├── labeled_data/       # 已标注数据
│   └── datasets/           # 训练/测试数据集
│
├── src/
│   ├── wave_extractor.py   # 波形提取模块
│   ├── annotation_tool/    # 标注工具界面
│   │   ├── gui.py          # 图形界面
│   │   └── cli.py          # 命令行界面
│   ├── data_processor.py   # 数据处理
│   ├── model_trainer.py    # 模型训练
│   └── evaluator.py        # 模型评估
│
├── models/                 # 训练好的模型
│   ├── pretrained/         # 预训练模型
│   └── trained/            # 自定义训练模型
│
├── notebooks/              # Jupyter笔记本
│   ├── data_exploration.ipynb
│   └── model_testing.ipynb
│
└── docs/                   # 文档
    ├── user_guide.md       # 用户指南
    └── developer_guide.md  # 开发者指南
```

```markdown
## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 提取音频波形：
```bash
python src/wave_extractor.py -i data/raw_audio/ -o data/extracted_waves/
```

3. 启动标注工具：
```bash
python src/annotation_tool/gui.py
```

4. 训练模型：
```bash
python src/model_trainer.py --data data/labeled_data/ --output models/trained/
```

## 数据流程

1. **波形提取**：从长音频中分割出有意义的波形片段
2. **人工标注**：通过GUI工具为波形打标签
3. **模型训练**：使用标注数据训练分类器
4. **性能评估**：测试模型在新数据上的表现

## 技术栈

- 音频处理：librosa, pydub
- 数据标注：PyQt/PySide 或 Tkinter
- 机器学习：scikit-learn, TensorFlow/PyTorch
- 可视化：matplotlib, seaborn

## 贡献指南

欢迎提交Pull Request！请先阅读[贡献指南](docs/contributing.md)。

## 许可证

MIT License
```

## 核心代码实现建议

### 1. 波形提取 (`wave_extractor.py`)
```python
import librosa
import numpy as np
import os

def extract_waveforms(input_dir, output_dir, segment_length=5.0, sr=22050):
    """从音频文件中提取固定长度的波形片段"""
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.endswith('.wav') or file.endswith('.mp3'):
            path = os.path.join(input_dir, file)
            y, sr = librosa.load(path, sr=sr)
            
            # 计算分段数
            samples_per_segment = int(segment_length * sr)
            n_segments = len(y) // samples_per_segment
            
            for i in range(n_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]
                
                # 保存波形片段
                np.save(os.path.join(output_dir, f"{file[:-4]}_seg{i}.npy"), segment)
```

### 2. 标注工具界面 (`annotation_tool/gui.py`)
```python
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
```

### 3. 模型训练 (`model_trainer.py`)
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def load_labeled_data(data_dir):
    """加载已标注的波形数据和标签"""
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            wave = np.load(os.path.join(data_dir, file))
            label = ...  # 从配套的标签文件读取
            X.append(wave)
            y.append(label)
    return np.array(X), np.array(y)

def extract_features(waveforms, sr=22050):
    """从波形中提取特征"""
    features = []
    for wave in waveforms:
        # 示例特征：MFCC系数
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13)
        features.append(mfcc.mean(axis=1))
    return np.array(features)

def train_model(X_train, y_train):
    """训练分类模型"""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # 加载数据
    X, y = load_labeled_data("data/labeled_data/")
    
    # 特征提取
    X_features = extract_features(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    joblib.dump(model, "models/trained/wave_classifier.pkl")
```

这个项目结构提供了完整的音频标注和机器学习工作流，您可以根据实际需求调整各个模块的实现细节。