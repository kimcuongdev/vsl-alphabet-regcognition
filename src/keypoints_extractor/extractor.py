import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get_ltrb(self):
        return self.x_min, self.y_min, self.x_max, self.y_max


class HandSkeletonExtractor:
    """
    Extractor nhận vào một frame BGR, trả ra:
        - frame_with_overlay: frame gốc có vẽ bbox + keypoint
        - skeleton_img: ảnh 224x224, nền đen, vẽ graph keypoint màu trắng
    """

    def __init__(
        self,
        target_size: int = 224,
        margin: int = 20,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        line_thickness: int = 2,  # độ dày của graph (line)
        circle_radius: int = 5,  # bán kính của các điểm (node)
        line_color: tuple = (255, 255, 255),
        circle_color: tuple = (0, 0, 255),
        channels: int = 1,
    ):
        self.target_size = target_size
        self.margin = margin
        self.line_thickness = line_thickness
        self.circle_radius = circle_radius
        self.line_color = line_color
        self.circle_color = circle_color
        self.channels = channels

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Khởi tạo MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # DrawingSpec cho frame gốc (keypoint + graph)
        self.drawing_spec_landmark = self.mp_drawing.DrawingSpec(
            color=self.circle_color,  # trắng
            thickness=self.line_thickness,
            circle_radius=self.circle_radius,
        )
        self.drawing_spec_connection = self.mp_drawing.DrawingSpec(
            color=self.line_color,  # trắng
            thickness=self.line_thickness,
        )

    def _get_frame_landmark(self, frame):
        """
        Trả về hand_landmarks nếu detect được tay
        Ngược lại trả về None
        """
        if frame is None:
            print("Input frame is None.")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return hand_landmarks
        return None

    def _get_hand_bbox(self, frame):
        """
        Trả về bbox của khung xương bàn tay dưới dạng
        (x_min, y_min, x_max, y_max)
        """
        hand_landmarks = self._get_frame_landmark(frame)
        if hand_landmarks is None:
            return None
        h, w, _ = frame.shape

        x_min = int(min([lm.x * w for lm in hand_landmarks.landmark]))
        x_max = int(max([lm.x * w for lm in hand_landmarks.landmark]))
        y_min = int(min([lm.y * h for lm in hand_landmarks.landmark]))
        y_max = int(max([lm.y * h for lm in hand_landmarks.landmark]))

        x_min = max(0, x_min - self.margin)
        y_min = max(0, y_min - self.margin)
        x_max = min(w, x_max + self.margin)
        y_max = min(h, y_max + self.margin)

        if x_max <= x_min or y_max <= y_min:
            return None

        return BoundingBox(x_min, y_min, x_max, y_max)

    def _extract_info(self, frame):
        hand_landmarks = self._get_frame_landmark(frame)
        bbox = self._get_hand_bbox(frame)
        return frame, hand_landmarks, bbox

    def _get_keypoints_img(self, frame):
        """
        Trả về ảnh khung xương của bàn tay trong frame
        (vẽ giống hệt annotated_img nhưng nền đen, resize về target_size x target_size)
        """

        frame, hand_landmarks, bbox = self._extract_info(frame)
        H, W, _ = frame.shape

        if bbox is None or hand_landmarks is None:
            return None

        x_min, y_min, x_max, y_max = bbox.get_ltrb()
        crop_h, crop_w = (y_max - y_min), (x_max - x_min)
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        points = []
        for lm in hand_landmarks.landmark:
            abs_x = lm.x * W
            abs_y = lm.y * H
            rel_x = (abs_x - x_min) / crop_w * self.target_size
            rel_y = (abs_y - y_min) / crop_h * self.target_size
            points.append((int(rel_x), int(rel_y)))

        for conn in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = conn
            x1, y1 = points[start_idx]
            x2, y2 = points[end_idx]
            cv2.line(canvas, (x1, y1), (x2, y2), self.line_color, self.line_thickness)

        for cx, cy in points:
            cv2.circle(canvas, (cx, cy), self.circle_radius, self.circle_color, -1)

        if self.channels == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            canvas = np.expand_dims(canvas, axis=-1)

        return canvas

    def _get_annotated_frame(self, frame):
        """
        Trả về frame có vẽ bbox
        """
        hand_landmarks = self._get_frame_landmark(frame)
        if hand_landmarks is None:
            return frame
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.drawing_spec_landmark,
            self.drawing_spec_connection,
        )
        return annotated_frame

    def process_frame(
        self,
        frame,
        frame_idx: int = 0,
        show_annotated=False,
        show_keypoints_image=False,
        inference_mode=False,
        save=False,
        save_path=None,
        save_name=None,
    ):
        """
        Xử lý frame RGB, trả về:
            - annotated_frame: frame có vẽ bbox + keypoints
            - keypoints_img: ảnh khung xương bàn tay
        """
        _, _, bbox = self._extract_info(frame)
        # print(
        #     "bbox detect in process_frame:", None if bbox is None else bbox.get_ltrb()
        # )
        keypoints_img = self._get_keypoints_img(frame)
        annotated_frame = self._get_annotated_frame(frame)
        if show_annotated:
            cv2.imshow("Annotated Frame", annotated_frame)
        if show_keypoints_image:
            if keypoints_img is not None:
                cv2.imshow("Hand Skeleton", keypoints_img)
            else:
                cv2.imshow(
                    "Hand Skeleton",
                    np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8),
                )
        if save and save_path is not None and keypoints_img is not None:
            filename = f"{save_name}_{frame_idx:05d}.jpeg"
            full_path = os.path.join(save_path, filename)

            # Encode ảnh trước (OpenCV vẫn encode được)
            ok, encoded = cv2.imencode(".jpeg", keypoints_img)
            if not ok:
                print(f"[ERROR] Failed to encode image for frame {frame_idx}")
                return

            # Lưu bằng numpy (hỗ trợ Unicode)
            try:
                encoded.tofile(full_path)
                print(f"[OK] Saved: {full_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save {full_path}: {e}")
        if inference_mode:
            # annotated_frame = cv2.flip(cv2.resize(annotated_frame, (1280, 720)), 1)
            # frame_display = cv2.flip(cv2.resize(frame, (1280, 720)), 1)
            # bbox = self._get_hand_bbox(
            #     frame_display,
            #     # self._get_frame_landmark(frame_display),
            # )
            # print(bbox)
            return keypoints_img, annotated_frame, bbox

    def close(self):
        self.hands.close()
