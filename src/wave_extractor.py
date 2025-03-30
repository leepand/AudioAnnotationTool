
## 核心代码实现建议

### 1. 波形提取 (`wave_extractor.py`)

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