import cv2

img_path = r"dataset\a\hand4_a_bot_seg_5_cropped.jpeg"

from src.keypoints_extractor.extractor import HandSkeletonExtractor

extractor = HandSkeletonExtractor(target_size=400)
frame = cv2.imread(img_path)
cv2.imshow("Input Image", frame)
cv2.waitKey(0)
# extractor.process_frame(
#     frame,
#     show_annotated=True,
#     show_keypoints_image=True,
#     save=True,
#     save_path="/tmp",
#     save_name="test_hand",
#     frame_idx=0,
# )
