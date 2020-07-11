import sys
import os
from RetinaFace.retinaface import RetinaFace
import numpy as np
from skimage import transform as trans
import cv2
from imutils.video import FPS


class FaceDetector():
    def __init__(self, model_retina_path, gpu_id):
        self.model = RetinaFace(
            model_retina_path, 0, ctx_id=gpu_id, network='net3')        

    def detect(self, img, scale_ratio=1.0):
        ret = self.model.detect(
            img, 0.5, scales=[scale_ratio], do_flip=False)
        if ret is None:
            return [], []
        bboxes, points = ret
        if len(bboxes) == 0:
            return [], []
        return np.asarray(bboxes) , points

    def align(self, img, bbox=None, landmark=None, **kwargs):
        if isinstance(img, str):
            img = read_image(img, **kwargs)
        M = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size)>0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size)==1:
                image_size = [image_size[0], image_size[0]]
                assert len(image_size)==2
            assert image_size[0]==112
            assert image_size[0]==112 or image_size[1]==96
        if landmark is not None:
            assert len(image_size)==2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041] ], dtype=np.float32 )
            if image_size[1]==112:
                src[:,0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
            #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        if M is None:
            if bbox is None: #use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if len(image_size)>0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret 
        else: #do align using landmark
            assert len(image_size)==2
            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
            return warped


if __name__ == "__main__":
    import anyconfig
    import munch
    from image_graber import ImageGraber
    import cv2
    opt = anyconfig.load("settings.yaml")
    opt = munch.munchify(opt)

    detector = FaceDetector(opt.face_detector.model_path, 0)
    imageGraber = ImageGraber(opt.camera.url, opt.camera.fps, opt.camera.push2queue_freq, opt.camera.rgb)
    imageGraber.start()
    fps = FPS().start()
    while not imageGraber.stop:
        image = imageGraber.get_frame()
        bboxes, points = detector.detect(image, 1.0)
        # Comment this stuff if you want to test FPS
        aligned_faces = []
        for i, bbox in enumerate(bboxes):
            conf_score = bbox[4]
            coords_box = [int(val) for val in bbox[:4]]
            x_min, y_min, x_max, y_max = coords_box
            for point in points[i]:
                cv2.circle(image, (point[0], point[1]), 1, (0, 255, 0), 3)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            aligned_face = detector.align(image, coords_box, points[i], image_size='112,112')
            aligned_faces.append(aligned_face)
        if len(aligned_faces) > 0:
            stacked = np.hstack(aligned_faces)
            cv2.imshow('faces', stacked)

        cv2.imshow('cam', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:
            cv2.imwrite('output.jpg', image)
        fps.update()
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    imageGraber.dispose()
    imageGraber.join()
    cv2.destroyAllWindows()
