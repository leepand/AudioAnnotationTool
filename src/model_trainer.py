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