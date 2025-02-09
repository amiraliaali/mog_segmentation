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
    has_background = False
    has_foreground = False
    for i in bounding_boxes:
        if i[1] == "Foreground":
            has_foreground = True
        if i[1] == "Background":
            has_background = True
    assert has_background and has_foreground, "You need to define bounding boxes for both background and foreground"

def segment_image(image, mog):
    """ Assigns each pixel to the most likely Gaussian component and creates a binary mask. """
    h, w, _ = image.shape
    data = image.reshape((-1, 3))  # Flatten image into (num_pixels, 3)

    responsibilities = np.zeros((len(data), 2))  # Two components: foreground and background
    for k in range(2):
        responsibilities[:, k] = mog.weights[k] * mog.gaussian_pdf(data, mog.means[k], mog.covs[k])

    # Assign each pixel to the component with higher probability
    labels = np.argmax(responsibilities, axis=1)
    mask = labels.reshape(h, w)  # Reshape back to image dimensions

    return mask


if __name__ == "__main__":
    image = load_image(sys.argv[1])
    bounding_boxes = define_bg.run_box_drawer(image)
    # bounding_boxes = [((113, 408, 407, 188), 'Foreground'), ((544, 451, 254, 117), 'Foreground'), ((-32, 56, 1083, 337), 'Background'), ((-14, 417, 112, 374), 'Background'), ((100, 607, 979, 211), 'Background'), ((822, 333, 233, 251), 'Background')]

    assert_if_no_background_foreground(bounding_boxes)

    print(bounding_boxes)

    k_mean = KMeansInitialization(image, bounding_boxes)
    table_of_data = k_mean.run()

    mog = MoG(table_of_data)
    mog.run()

    mask = segment_image(image, mog)
    binary_mask = (mask * 255).astype(np.uint8)
    cv2.imwrite("segmented_mask.png", binary_mask)

    post_processed_mask = post_processing(binary_mask)
    cv2.imwrite("segmented_post_processed_mask.png", post_processed_mask)

    segmented_image = image.copy()
    segmented_image[post_processed_mask == 0] = [0, 0, 0]

    cv2.imwrite("segmented_image.png", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print("Segmentation complete. Saved mask and segmented image.")
