import sys
import define_bg
import cv2

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


if __name__ == "__main__":
    image = load_image(sys.argv[1])
    bounding_boxes = define_bg.run_box_drawer(image)

    assert_if_no_background_foreground(bounding_boxes)

    print(bounding_boxes)
