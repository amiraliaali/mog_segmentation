import sys
import define_bg
import cv2
from mog import MoG
from k_means_initialization import KMeansInitialization
import numpy as np
from post_processing import post_processing

MAX_WIDTH = 1200
MAX_HEIGHT = 800

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def assert_if_no_background_foreground(bounding_boxes):
    has_background = any(label == "Background" for _, label in bounding_boxes)
    has_foreground = any(label == "Foreground" for _, label in bounding_boxes)
    assert has_background and has_foreground, "You need to define bounding boxes for both background and foreground"

def segment_image(image, mog):
    h, w, _ = image.shape
    data = image.reshape((-1, 3))
    responsibilities = np.zeros((len(data), 2))
    for k in range(2):
        responsibilities[:, k] = mog.weights[k] * mog.gaussian_pdf(data, mog.means[k], mog.covs[k])
    labels = np.argmax(responsibilities, axis=1)
    mask = labels.reshape(h, w)
    return mask

def process_video(video_path, mog, output_path="segmented_output.mov"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = segment_image(frame, mog)
        binary_mask = (mask * 255).astype(np.uint8)
        # post_processed_mask = post_processing(binary_mask)
        segmented_frame = frame.copy()
        segmented_frame[binary_mask == 0] = [0, 0, 0]

        out.write(cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print(f"Video segmentation complete. Saved as {output_path}")

import cv2
import sys

def get_first_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_first, first_frame = cap.read()

    # Read the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret_last, last_frame = cap.read()

    cap.release()

    if not ret_first or not ret_last:
        raise ValueError("Could not read frames from video")
    
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

    return first_frame, last_frame


if __name__ == "__main__":
    # image = load_image(sys.argv[1])
    image_1, image_2 = get_first_last_frame(sys.argv[2])

    bounding_boxes = define_bg.run_box_drawer(image_1)
    # bounding_boxes.extend(define_bg.run_box_drawer(image_2))
    assert_if_no_background_foreground(bounding_boxes)

    k_mean = KMeansInitialization(image_1, bounding_boxes)
    table_of_data = k_mean.run()
    mog = MoG(table_of_data)
    mog.run()

    mask = segment_image(image_1, mog)
    binary_mask = (mask * 255).astype(np.uint8)
    cv2.imwrite("segmented_mask.png", binary_mask)

    post_processed_mask = post_processing(binary_mask)
    cv2.imwrite("segmented_post_processed_mask.png", post_processed_mask)

    segmented_image = image_1.copy()
    segmented_image[post_processed_mask == 0] = [0, 0, 0]
    cv2.imwrite("segmented_image.png", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print("Segmentation complete. Saved mask and segmented image.")

    if len(sys.argv) > 2:
        process_video(sys.argv[2], mog)