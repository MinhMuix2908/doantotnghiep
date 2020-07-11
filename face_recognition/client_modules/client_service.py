
from uuid import getnode as get_mac
import threading
import cv2
from config import cfg
from queue import Queue
from image_graber import ImageGraber
from face_detection import FaceDetector
import rpyc
import time
import numpy as np


rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

def cvtNetrefToNumpy(obj):
    obj = rpyc.classic.obtain(obj)
    return obj

CURRENT_TIME_IN_MILLISECOND = lambda: int(round(time.time() * 1000000))

class SendingService(threading.Thread):
    def __init__(self, server_conn, send_queue, detector, device_id=get_mac()):
        """
            Sending face information to server
            args:
                server_conn: Rpyc connection to server
                send_queue: (Queue) Sending queue
                device_id: Device ID
        """
        threading.Thread.__init__(self)
        self.server_conn = server_conn
        self.send_queue = send_queue
        self.detector = detector
        self.device_id = device_id
        self.running = True


    def send(self, image, scaling_ratio=0.5):
        """
            Detect and align face then send it to AI Server
            args:
                image: Image captured from camera
        """
        #Get bboxes and face landmarks
        bboxes, points = self.detector.detect(image, scaling_ratio)
        face_images = []
        text_positions = []
        
        for i, bbox in enumerate(bboxes):
            conf_score = bbox[4]
            if conf_score < cfg.face_detector.confident_score_threshold:

                continue
            coords_box = [int(val) for val in bbox[:4]]
            text_pos = np.array([coords_box[0], coords_box[1] - 15 if coords_box[1] - 15 > 15 else coords_box[1] + 15])
            text_positions.append(text_pos)
            #align face
            aligned_face = self.detector.align(image, coords_box, points[i], image_size='112,112')

            # Use timestamp for mapping with face
            image = cv2.rectangle(image, (coords_box[0], coords_box[1]), (coords_box[2], coords_box[3]), (255, 0, 0), 2)
            


            face_images.append(aligned_face)
        face_images = np.asarray(face_images, dtype=np.uint8)
        text_positions = np.asarray(text_positions, dtype=np.int)
        # print(face_images.shape)
        ts = str(CURRENT_TIME_IN_MILLISECOND())
        resized_image = cv2.resize(image.copy(), (600, 480), interpolation=cv2.INTER_NEAREST)
        #Wrap information and send to the server
        # self.server_conn.recognize(self.device_id, resized_image.copy(), face_images.copy(), text_positions.copy(), ts)
        self.server_conn.recognize(self.device_id, resized_image.tostring(), resized_image.shape, face_images.tostring(), face_images.shape, text_positions.tostring(), text_positions.shape, ts)

    def run(self):
        while self.running:
            if self.send_queue.qsize() == 0:
                # time.sleep(0.1)
                continue
            while self.send_queue.qsize() > 10:
                self.send_queue.get()
            image = self.send_queue.get()
            self.send(image)           

    def dispose(self):
        self.running = False

class ReceivingService(threading.Thread):
    def __init__(self, server_conn, receive_queue, device_id=get_mac()):
        """
            Receiving results from server
            args:
                server_conn: Rpyc connection to server
                receive_queue: Result queue
                device_id: Device ID
        """
        threading.Thread.__init__(self)
        self.server_conn = server_conn
        self.receive_queue = receive_queue
        self.device_id = device_id
        self.running = True

    def run(self):
        while self.running:
            ret ,return_value = self.server_conn.get_result(self.device_id)
            if ret == False:
                continue
            self.receive_queue.put(return_value)

    def dispose(self):
        self.running = False
    

class ClientService:
    def __init__(self):
        self.device_id = get_mac()

        c = rpyc.connect(cfg.server.ip, cfg.server.port, config = {"allow_public_attrs" : True})
        self.server = c.root

        self.detector = FaceDetector(cfg.face_detector.model_path, -1)
        self.sending_queue = Queue()
        self.sending_service = SendingService(self.server, self.sending_queue, self.detector)

        self.receiving_queue = Queue()
        # self.receiving_service = ReceivingService(self.server, self.receiving_queue)

        self.image_graber = ImageGraber(cfg.camera.url, cfg.camera.fps, cfg.camera.push2queue_freq, cfg.camera.rgb)
        
        self.server.register_device(self.device_id)
        self.image_graber.start()

    def __init_recognize_service__(self):
        self.sending_service.start()
        # self.receiving_service.start()        

    def run_register_new_face(self, name):
        """
            Register new face to database
            Args:
                name: Name
        """        
        while not self.image_graber.stop:
            image = self.image_graber.get_frame()
            image_viz = image.copy()
            bboxes, points = self.detector.detect(image, 0.5)
            aligned_face = None

            #Make sure that captured image contains only one face.
            if len(bboxes) != 1:
                cv2.putText(image_viz, 'No face founded', (0, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255) , 2, cv2.LINE_AA) 
                cv2.putText(image_viz, 'or there is more than one face', (0, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255) , 2, cv2.LINE_AA) 
            elif bboxes[0][4] < cfg.face_detector.confident_score_threshold: #conf score
                pass
            else:
                bbox = bboxes[0]
                i = 0

                #Convert to int
                coords_box = [int(val) for val in bbox[:4]]
                x_min, y_min, x_max, y_max = coords_box

                #Draw faceboxes and landmarks
                for point in points[i]:
                    cv2.circle(image_viz, (point[0], point[1]), 1, (0, 255, 0), 3)
                cv2.rectangle(image_viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

                #Align face
                aligned_face = self.detector.align(image, coords_box, points[i], image_size='112,112')

            #Visualize
            if aligned_face is not None:
                cv2.imshow('cutted', aligned_face)
            cv2.imshow('camera', image_viz)

            k = cv2.waitKey(1)
            if k == ord('q'): #Press q to exits
                self.image_graber.dispose()
                self.image_graber.join()
            elif k == 32 and aligned_face is not None: #Press space to add new face to db
                self.server.add_new_face(aligned_face, name)     
                break           

    def run_demo(self):
        """
            Run demo Face Recognition
        """
        self.__init_recognize_service__()
        while True:
            image = self.image_graber.get_frame()
            self.sending_queue.put(image)
            # if self.receiving_queue.qsize() > 0:             
            #     returned_image = cvtNetrefToNumpy(self.receiving_queue.get())
            #     cv2.imshow('ret', returned_image)
            # cv2.imshow('camera', image)
            # if cv2.waitKey(20) & 0xFF == ord('q'):
            #     self.dispose()
            #     break
            time.sleep(0.02)

    def dispose(self):
        """
            Stop threads and clean up memories 
        """
        self.image_graber.dispose()
        self.sending_service.dispose()
        self.receiving_service.dispose()
        if self.image_graber.isAlive():
            self.image_graber.join()
        if self.sending_service.isAlive():
            self.sending_service.join()
        if self.receiving_service.isAlive():
            self.receiving_service.join()
