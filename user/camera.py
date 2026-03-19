"""
摄像头管理 + MediaPipe Pose 姿态检测模块
使用 mp.solutions.pose（需 mediapipe<=0.10.14），
提供视图分类、站位验证、自动拍照和 MJPEG 流生成。
"""

import time
import threading
from typing import Optional
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

VIEW_ORDER = ["front", "right", "back", "left"]
VIEW_LABELS = {
    "front": "正面", "right": "右侧",
    "back": "背面", "left": "左侧", "unknown": "未知",
}
VIEW_LABELS_EN = {
    "front": "Front", "right": "Right",
    "back": "Back", "left": "Left", "unknown": "?",
}


def _voice_turn_hint(required_view: str) -> str:
    """需要某视图时，播报用户应转向的方向（需右视图=用户向左转）。"""
    if required_view == "right":
        return "请向左转身，让摄像头拍到您的右侧"
    if required_view == "left":
        return "请再向左转身，让摄像头拍到您的左侧"
    if required_view == "front":
        return "请面向摄像头"
    if required_view == "back":
        return "请背对摄像头"
    return f"请转向{VIEW_LABELS.get(required_view, required_view)}"


SIDE_THRESHOLD = 0.04
FRONT_BACK_THRESHOLD = 0.03
STABLE_FRAMES_REQUIRED = 20
STABLE_HOLD_SECONDS = 2.0
FOOT_REGION_TOP = 0.75


class ViewClassifier:
    """根据 MediaPipe Pose 关键点判断用户面朝方向。阈值随身体在画面中的尺度自适应，避免远距离时误判为正面。"""

    def classify(self, landmarks) -> str:
        left_eye, right_eye = landmarks[2], landmarks[5]
        left_ear, right_ear = landmarks[7], landmarks[8]
        left_shoulder, right_shoulder = landmarks[11], landmarks[12]

        shoulder_span = abs(right_shoulder.x - left_shoulder.x)
        scale = max(shoulder_span, 0.08)
        side_thresh = max(0.012, min(0.055, SIDE_THRESHOLD * (scale / 0.25)))
        fb_thresh = max(0.01, min(0.04, FRONT_BACK_THRESHOLD * (scale / 0.25)))

        eye_center_x = (left_eye.x + right_eye.x) / 2
        ear_center_x = (left_ear.x + right_ear.x) / 2
        eye_ear_diff = eye_center_x - ear_center_x

        if abs(eye_ear_diff) > side_thresh:
            return "right" if eye_ear_diff < 0 else "left"

        shoulder_diff = right_shoulder.x - left_shoulder.x
        if shoulder_diff > fb_thresh:
            return "back"
        elif shoulder_diff < -fb_thresh:
            return "front"

        return "unknown"


class StandingValidator:
    """验证用户脚部是否在画面底部指定区域。"""

    def validate(self, landmarks) -> bool:
        foot_indices = [27, 28, 29, 30, 31, 32]
        visible_count = 0
        for idx in foot_indices:
            pt = landmarks[idx]
            if pt.visibility < 0.4:
                continue
            visible_count += 1
            if pt.y < FOOT_REGION_TOP:
                return False
        return visible_count >= 2


class CameraManager:
    """管理 OpenCV 摄像头和 MediaPipe 姿态检测的核心类。"""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._pose = None
        self._lock = threading.Lock()
        self._running = False

        self.view_classifier = ViewClassifier()
        self.standing_validator = StandingValidator()

        self.current_view = "unknown"
        self.standing_ok = False
        self._stable_history: deque = deque(maxlen=STABLE_FRAMES_REQUIRED)
        self._last_frame: Optional[np.ndarray] = None
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_landmarks = None

        self.captured_views: dict[str, Optional[np.ndarray]] = {v: None for v in VIEW_ORDER}
        self._capture_order_idx = 0
        self._last_voice_time: dict[str, float] = {}
        self._pending_voice: Optional[str] = None
        self._capture_ready_at: Optional[float] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def next_required_view(self) -> Optional[str]:
        if self._capture_order_idx >= len(VIEW_ORDER):
            return None
        return VIEW_ORDER[self._capture_order_idx]

    @property
    def all_views_captured(self) -> bool:
        return self._capture_order_idx >= len(VIEW_ORDER)

    def start(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._running = True
        self._capture_order_idx = 0
        self.captured_views = {v: None for v in VIEW_ORDER}
        self._stable_history.clear()
        self._capture_ready_at = None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._pose:
            self._pose.close()
            self._pose = None

    def reset_capture(self):
        self._capture_order_idx = 0
        self.captured_views = {v: None for v in VIEW_ORDER}
        self._stable_history.clear()
        self._capture_ready_at = None

    def _throttled_voice(self, key: str, text: str, interval: float = 4.0) -> Optional[str]:
        now = time.time()
        last = self._last_voice_time.get(key, 0)
        if now - last >= interval:
            self._last_voice_time[key] = now
            return text
        return None

    def process_frame(self) -> Optional[np.ndarray]:
        if not self._running or not self._cap:
            return None
        if not self._lock.acquire(timeout=0.5):
            return self._last_frame
        try:
            return self._process_frame_locked()
        finally:
            self._lock.release()

    def _process_frame_locked(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        self._last_raw_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        self._pending_voice = None
        required = self.next_required_view

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self._last_landmarks = landmarks

            self.current_view = self.view_classifier.classify(landmarks)
            self.standing_ok = self.standing_validator.validate(landmarks)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            self._draw_guide_overlay(frame, required)

            if required:
                self._stable_history.append(self.current_view)
                stable_count = sum(1 for v in self._stable_history if v == required)

                if self.current_view != required or not self.standing_ok:
                    self._capture_ready_at = None
                if stable_count >= STABLE_FRAMES_REQUIRED and self.standing_ok:
                    now = time.time()
                    if self._capture_ready_at is None:
                        self._capture_ready_at = now
                    elif now - self._capture_ready_at >= STABLE_HOLD_SECONDS:
                        self._do_auto_capture(required)
                        self._capture_ready_at = None
                elif self.current_view != required:
                    self._pending_voice = self._throttled_voice(
                        f"guide_{required}", _voice_turn_hint(required)
                    )
                elif not self.standing_ok:
                    self._pending_voice = self._throttled_voice(
                        "standing", "请站在画面中央，确保全身可见"
                    )
        else:
            self.current_view = "unknown"
            self.standing_ok = False
            self._last_landmarks = None
            self._stable_history.clear()
            self._pending_voice = self._throttled_voice(
                "no_pose", "未检测到人体，请站到摄像头前"
            )
            self._draw_guide_overlay(frame, required)

        self._draw_status_bar(frame, required)
        self._last_frame = frame
        return frame

    def _do_auto_capture(self, view: str):
        if self._last_raw_frame is not None:
            self.captured_views[view] = self._last_raw_frame.copy()
            self._capture_order_idx += 1
            self._stable_history.clear()

            next_view = self.next_required_view
            if next_view:
                self._pending_voice = f"{VIEW_LABELS[view]}已拍摄，{_voice_turn_hint(next_view)}"
            else:
                self._pending_voice = "四个视图全部拍摄完成"

    def manual_capture(self) -> Optional[str]:
        required = self.next_required_view
        if required and self._last_raw_frame is not None:
            self.captured_views[required] = self._last_raw_frame.copy()
            self._capture_order_idx += 1
            self._stable_history.clear()
            return required
        return None

    def _draw_guide_overlay(self, frame: np.ndarray, required_view: Optional[str]):
        h, w = frame.shape[:2]
        foot_y = int(h * FOOT_REGION_TOP)
        cv2.line(frame, (0, foot_y), (w, foot_y), (0, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "--- foot zone ---", (w // 2 - 60, foot_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1, cv2.LINE_AA)

        cx, cy = w // 2, h // 2
        box_w, box_h = int(w * 0.4), int(h * 0.8)
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1, cv2.LINE_AA)

    def _draw_status_bar(self, frame: np.ndarray, required_view: Optional[str]):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        view_text = f"View: {VIEW_LABELS_EN.get(self.current_view, '?')}"
        color = (0, 255, 0) if self.current_view == required_view else (0, 165, 255)
        cv2.putText(frame, view_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        stand_text = "Stand: OK" if self.standing_ok else "Stand: --"
        stand_color = (0, 255, 0) if self.standing_ok else (0, 0, 255)
        cv2.putText(frame, stand_text, (200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, stand_color, 2, cv2.LINE_AA)

        if required_view:
            req_text = f"Need: {VIEW_LABELS_EN.get(required_view, '?')}"
            cv2.putText(frame, req_text, (380, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        progress = f"{self._capture_order_idx}/{len(VIEW_ORDER)}"
        cv2.putText(frame, progress, (w - 70, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    def get_jpeg_bytes(self) -> Optional[bytes]:
        frame = self.process_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def get_pose_status(self) -> dict:
        required = self.next_required_view
        stable_count = (
            sum(1 for v in self._stable_history if v == required) if required else 0
        )
        return {
            "view": self.current_view,
            "standing_ok": self.standing_ok,
            "stable_count": stable_count,
            "required_view": required,
            "captured": {v: (self.captured_views[v] is not None) for v in VIEW_ORDER},
            "all_done": self.all_views_captured,
            "capture_index": self._capture_order_idx,
        }

    def pop_voice_message(self) -> Optional[str]:
        msg = self._pending_voice
        self._pending_voice = None
        return msg

    def get_captured_image_jpeg(self, view: str) -> Optional[bytes]:
        img = self.captured_views.get(view)
        if img is None:
            return None
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    def get_captured_image_png(self, view: str) -> Optional[bytes]:
        img = self.captured_views.get(view)
        if img is None:
            return None
        _, buf = cv2.imencode('.png', img)
        return buf.tobytes()
