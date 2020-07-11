import cv2
import threading
from queue import Queue
import time

class ImageGraber(threading.Thread):
    def __init__(self, url, fps, push2queue_freq, output_format):
        threading.Thread.__init__(self)
        self.cam = cv2.VideoCapture(url)
        #self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        print("CAM FPS = {}".format(self.fps))
        #assert int(self.fps) == fps, "ERROR: Cannot set FPS successfully"
        self.raw_queue = Queue()
        self.rgb_output = output_format
        self.push2queue_freq = push2queue_freq
        self.stop = False


    def run(self):
        current_frame_idx = 0
        waiting_frame_time = round(0.5 / self.fps, 3)
        
        while True:
            if self.stop:
                break
            
            ret, frame = self.cam.read()
            if ret == False:
                time.sleep(waiting_frame_time)
                continue
            if current_frame_idx % self.push2queue_freq == 0:                
                # if self.rgb_output is True:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.raw_queue.put(frame)
                current_frame_idx = 0
            
            current_frame_idx += 1

            if self.raw_queue.qsize() >= 10:
                self.raw_queue.get() # Prevent the size of the queue becoming too large

    def get_frame(self):
        return(self.raw_queue.get())
    
    def dispose(self):
        self.stop = True
        time.sleep(1) # prevent Segmentation fault (core dumped) when Camera tried to read frame
        self.cam.release()

if __name__ == "__main__":    
    import anyconfig
    import munch
    opt = anyconfig.load("settings.yaml")
    opt = munch.munchify(opt)
    imageGraber = ImageGraber(opt.camera.url, opt.camera.fps, opt.camera.push2queue_freq, opt.camera.rgb)
    imageGraber.start()
    while not imageGraber.stop:
        image = imageGraber.get_frame()
        cv2.imshow('cam', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()