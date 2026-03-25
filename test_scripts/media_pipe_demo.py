import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import urllib.request
import os

# 下载模型文件
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# 21个关键点的连接关系（用于画骨架线）
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 拇指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 食指
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # 中指
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # 无名指
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 小指
    (5, 9),
    (9, 13),
    (13, 17),  # 掌心横向连接
]

# 左右手配色方案
#           (关键点颜色,     骨架线颜色,       文字颜色)
LEFT_HAND = ((255, 100, 100), (200, 60, 60), (255, 150, 150))  # 蓝色系（BGR）
RIGHT_HAND = ((100, 255, 100), (60, 200, 60), (150, 255, 150))  # 绿色系（BGR）

# 配置检测器
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape

        for hand_idx, hand in enumerate(result.hand_landmarks):
            # 判断左右手
            # MediaPipe 返回的 handedness 是从摄像头视角看的
            # 由于我们做了镜像翻转，所以 "Left" 实际对应用户的左手
            handedness = result.handedness[hand_idx][0]
            label = handedness.category_name  # "Left" 或 "Right"
            score = handedness.score

            # 镜像后：MediaPipe的"Left"是用户的右手，"Right"是用户的左手
            if label == "Left":
                dot_color, line_color, text_color = RIGHT_HAND
                display_label = "Right"  # 用户视角的右手
            else:
                dot_color, line_color, text_color = LEFT_HAND
                display_label = "Left"  # 用户视角的左手

            # 先把21个关键点转成像素坐标列表
            points = []
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            # 画骨架连接线
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], line_color, 2)

            # 画关键点 + 编号
            for idx, (cx, cy) in enumerate(points):
                # 关键点圆点
                cv2.circle(frame, (cx, cy), 5, dot_color, -1)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)  # 白色描边

                # 编号文字（稍微偏移避免遮挡圆点）
                cv2.putText(
                    frame,
                    str(idx),
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # 在手腕位置显示左/右手标签
            wrist_x, wrist_y = points[0]
            cv2.putText(
                frame,
                f"{display_label} ({score:.0%})",
                (wrist_x - 30, wrist_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                dot_color,
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
