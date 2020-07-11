from FaceDetection import FaceDetector
import cv2
import anyconfig
import munch

opt = anyconfig.load("settings.yaml")
opt = munch.munchify(opt)

detector = FaceDetector(opt.face_detector.model_path, -1)



image = cv2.imread('/home/minhbq/Desktop/people-14.jpg')
# image = image[:, :, ::-1]
print(image.shape)
bboxes, points = detector.detect(image, 0.2)
aligned_faces = []
print(bboxes)
for i, bbox in enumerate(bboxes):
    print(i)
    conf_score = bbox[4]
    coords_box = [int(val) for val in bbox[:4]]
    if conf_score < 0.5:
        continue
    x_min, y_min, x_max, y_max = coords_box
    for point in points[i]:
        print(image.shape)
        # cv2.circle(image, (point[0], point[1]), 1, (0, 255, 0), 3)
    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    aligned_face = detector.align(image, coords_box, points[i], image_size='112,112')
    aligned_faces.append(aligned_face)