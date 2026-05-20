import cv2
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# 1. 加载模型
model = YOLO("runs/classify/train4/weights/best.pt")

# 2. 类别名和风险权重
class_names = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
risk_map = {
    "c0": 0.0,
    "c1": 1.0,
    "c2": 1.0,
    "c3": 1.0,
    "c4": 1.0,
    "c5": 0.6,
    "c6": 0.5,
    "c7": 0.9,
    "c8": 0.8,
    "c9": 0.5,
}

# 3. 滑动窗口
window_size = 10
prob_window = deque(maxlen=window_size)

# 4. 读取视频
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

frame_id = 0
sample_interval = 3   # 每3帧取1帧
alert_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % sample_interval != 0:
        continue

    # 5. 单帧分类
    results = model(frame, verbose=False)
    probs = results[0].probs.data.cpu().numpy()   # 10类概率

    prob_window.append(probs)

    if len(prob_window) < window_size:
        continue

    # 6. 窗口概率平均
    avg_probs = np.mean(np.array(prob_window), axis=0)
    pred_idx = int(np.argmax(avg_probs))
    pred_class = class_names[pred_idx]

    # 7. 风险分数
    risk_score = 0.0
    for i, cls_name in enumerate(class_names):
        risk_score += avg_probs[i] * risk_map[cls_name]

    # 8. 报警逻辑
    if risk_score >= 0.6:
        alert_count += 1
    else:
        alert_count = 0

    alarm = alert_count >= 5

    print(f"Frame {frame_id}: class={pred_class}, risk={risk_score:.3f}, alarm={alarm}")

cap.release()