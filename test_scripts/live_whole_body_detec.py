import cv2
from rtmlib import Wholebody, draw_skeleton
from loguru import logger
from signlangagent.utils.device_utils import get_available_device

cap = cv2.VideoCapture(1)
wholebody = Wholebody(
    to_openpose=False,  # 改成 False
    backend="onnxruntime",
    device=get_available_device(),  # Mac 上用 mps 或 cpu，不支持 cuda
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    keypoints, scores = wholebody(frame)
    logger.debug(f"Keypoints.shape:\n{keypoints.shape}")
    logger.debug(f"Scores.shape:\n{scores.shape}")
    img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.43)

    cv2.imshow("video", img_show)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
