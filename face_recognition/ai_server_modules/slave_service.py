import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import rpyc
from queue import Queue
from face_recognitor import FaceRecognitor
from web_service import WebService
from frame_updater import FrameUpdater
import anyconfig
import munch
from utils import cvtNetrefToNumpy
from queue import Queue
import threading
import time
import cv2
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

class RecognizeService(threading.Thread):
    def __init__(self, face_recognitor, cfg):
        threading.Thread.__init__(self)
        self.face_recognitor = face_recognitor
        self.face_info_queue = Queue()
        self.result_dict = {}
        self.frame_dict = {}        
        self.frame_updater = FrameUpdater(self.frame_dict, self.result_dict)
        self.web_service = WebService(self.frame_dict, cfg)
        self.frame_updater.start()
        self.web_service.start()

    def add_new_face(self, new_face, name):
        new_face = cvtNetrefToNumpy(new_face)
        new_face = new_face[:,:,::-1][None, ...]
        print(f'Adding {name}')
        self.face_recognitor.register(new_face, name)
        print(f'Added {name} to db')

    def register_device(self, deviceID):
        if deviceID in self.result_dict.keys():
            self.result_dict[deviceID].queue.clear()
        print(deviceID)
        self.result_dict[deviceID] = Queue()
        print(self.result_dict.keys())
        return deviceID

    def recognize(self, deviceID, image, face_images, text_positions, ts):
        # cvt_time = time.time()
        # image = cvtNetrefToNumpy(image)
        deviceID = cvtNetrefToNumpy(deviceID)
        ts = cvtNetrefToNumpy(ts)
        # text_positions = cvtNetrefToNumpy(text_positions)
        # cvt_time = time.time() - cvt_time
        # print("Convert time:", cvt_time)
        self.face_info_queue.put({"deviceID": deviceID, "image": image, "face_images": face_images, "text_positions":  text_positions, "ts": ts})

    def get_result(self, deviceID):
        if deviceID not in self.result_dict.keys():
            return False, None
        while self.result_dict[deviceID].qsize() > 0:
            result = self.result_dict[deviceID].get()
            return True, result
        return False, None

    def run(self):
        while True:
            if self.face_info_queue.qsize() == 0:
                continue
            # process_time = time.time()
            while self.face_info_queue.qsize() > 5:
                self.face_info_queue.get()
            info = self.face_info_queue.get()
            # read_data_time = time.time()
            deviceID = info["deviceID"]
            image = info["image"]
            face_images = info["face_images"]
            ts = info["ts"]
            text_positions = info["text_positions"]
            # read_data_time = time.time() - read_data_time
            # print("Read_data_time:", read_data_time)
            # recognize_time = time.time()
            for i, face_image in enumerate(face_images):
                face_image = face_image[:, :, ::-1][None, ...]
                name = self.face_recognitor.recognize(face_image)
                # cv2.rectangle(image, (face_coord[0], face_coord[1]), (face_coord[2], face_coord[3]), (255, 0, 0), 2)
                cv2.putText(image, name, (text_positions[i][0], text_positions[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
            self.result_dict[deviceID].put(image)
            # recognize_time = time.time() - recognize_time
            # print("recognize_time:", recognize_time)
            # process_time = time.time() - process_time
            # print('Total processing time:', process_time)

class AIService(rpyc.Service):
    def __init__(self, recognizeService):
        self.recognizeService = recognizeService

    def exposed_add_new_face(self, face_image, name):
        self.recognizeService.add_new_face(face_image, name)

    def exposed_register_device(self, deviceID):
        self.recognizeService.register_device(deviceID)        

    def exposed_recognize(self, deviceID, image, image_shape, face_images, face_shape, text_positions, text_positions_shape, ts):
        # convert_img_time = time.time()
        image = np.fromstring(image, dtype=np.uint8)
        image = np.reshape(image, image_shape)

        face_images = np.fromstring(face_images, dtype=np.uint8)
        face_images = np.reshape(face_images, face_shape)
        
        text_positions = np.fromstring(text_positions, dtype=np.int)
        text_positions = np.reshape(text_positions, text_positions_shape)

        # convert_img_time = time.time() - convert_img_time
        # print('convert_img_time:', convert_img_time)
        self.recognizeService.recognize(deviceID, image, face_images, text_positions, ts)
        
    def exposed_get_result(self, deviceID):
        # return self.recognizeService.get_result(deviceID)
        pass

    def on_connect(self, conn):
        pass # TODO: nothing
        
    def on_disconnect(self, conn):
        pass # TODO: nothing

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    from rpyc.utils.helpers import classpartial
    import numpy as np
    cfg = anyconfig.load('settings.yaml')
    cfg = munch.munchify(cfg)

    face_recognitor = FaceRecognitor(cfg)
    face_recognitor.get_embeddings(np.ones((1,112,112,3)))

    recognizeService = RecognizeService(face_recognitor, cfg)
    recognizeService.start()

    service = classpartial(AIService, recognizeService)
    t = ThreadedServer(service, port=45685, protocol_config={"allow_public_attrs" : True})
    print("Ready to serve")
    t.start()