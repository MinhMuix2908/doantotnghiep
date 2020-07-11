import cv2
import os
import glob
import anyconfig
import munch
from face_detection import FaceDetector
import tqdm

DATA_PATH = "/hdd/DATA/TPA/face/recognition/datasets/casia_112x112"
OUTPUT_PATH = "/hdd/DATA/TPA/face/recognition/datasets/casia_112x112_aligned"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

print(len(os.listdir(DATA_PATH)))

opt = anyconfig.load("settings.yaml")
opt = munch.munchify(opt)
detector = FaceDetector(opt.face_detector.model_path, 0)


for id in tqdm.tqdm(os.listdir(DATA_PATH)):
    id_path = os.path.join(DATA_PATH, id)
    output_id_path = os.path.join(OUTPUT_PATH, id)
    if not os.path.exists(output_id_path):
        os.makedirs(output_id_path)
    for filename in os.listdir(id_path):
        file_path = os.path.join(id_path, filename)
        output_file_path = os.path.join(output_id_path, filename)
        image = cv2.imread(file_path)
        bboxes, points = detector.detect(image, 1.0)
        if len(bboxes) == 0:
            continue
        bbox = bboxes[0]
        point = points[0]
        conf_score = bbox[4]
        coords_box = [int(val) for val in bbox[:4]]
        if conf_score < 0.5:
            continue
        x_min, y_min, x_max, y_max = coords_box
        aligned_face = detector.align(image, coords_box, point, image_size='112,112')
        cv2.imwrite(output_file_path, aligned_face)
        # print(output_file_path)
    