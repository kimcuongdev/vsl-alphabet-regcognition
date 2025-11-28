import argparse
from pathlib import Path

import cv2

from src.keypoints_extractor.extractor import HandSkeletonExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="1 = grayscale, 3 = rgb",
    )
    parser.add_argument(
        "--output",
        type=int,
        default=224,
        help="kích thước cạnh ảnh output (square), ví dụ 224",
    )
    args = parser.parse_args()

    channels = args.channels
    target_size = args.output

    # map channels -> scale folder
    scale = "grayscale" if channels == 1 else "rgb"

    video_paths = [
        r"data\raw_video\A.mp4",
        r"data\raw_video\B.mp4",
        r"data\raw_video\C.mp4",
        r"data\raw_video\D.mp4",
        r"data\raw_video\Đ.mp4",
        r"data\raw_video\E.mp4",
        r"data\raw_video\G.mp4",
        r"data\raw_video\H.mp4",
        r"data\raw_video\I.mp4",
        r"data\raw_video\K.mp4",
        r"data\raw_video\L.mp4",
        r"data\raw_video\M.mp4",
        r"data\raw_video\mũ.mp4",
        r"data\raw_video\N.mp4",
        r"data\raw_video\O.mp4",
        r"data\raw_video\P.mp4",
        r"data\raw_video\Q.mp4",
        r"data\raw_video\R.mp4",
        r"data\raw_video\Râu.mp4",
        r"data\raw_video\S.mp4",
        r"data\raw_video\T.mp4",
        r"data\raw_video\U.mp4",
        r"data\raw_video\V.mp4",
        r"data\raw_video\X.mp4",
        r"data\raw_video\Y.mp4",
    ]

    for video_path in video_paths:
        video_path = Path(video_path)
        VIDEO_PATH = str(video_path)

        # label = tên file video không gồm đuôi, ví dụ A, Đ, mũ...
        label = video_path.stem

        # data/keypoint_img/{scale}/{output_size}/{label}
        SAVE_PATH = (
            Path("data") / "keypoint_img" / scale / str(target_size) / "v2" / label
        )
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Không mở được video: {VIDEO_PATH}, bỏ qua.")
            continue

        extractor = HandSkeletonExtractor(
            target_size=target_size,
            channels=channels,
        )

        frame_idx = 0
        video_title = video_path.stem  # dùng làm prefix khi lưu ảnh

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Hết video: {VIDEO_PATH}")
                break

            frame_idx += 1

            extractor.process_frame(
                frame=frame,
                frame_idx=frame_idx,
                show_annotated=True,
                show_keypoints_image=True,
                save=True,
                save_path=str(SAVE_PATH),
                save_name=video_title,  # ví dụ A_frame_001.jpeg
            )

            # Nhấn 'q' để thoát sớm
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        extractor.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
