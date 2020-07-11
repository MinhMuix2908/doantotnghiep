import threading
import time

class FrameUpdater(threading.Thread):
    def __init__(self, current_frame, result_dict):
        threading.Thread.__init__(self)
        self.result_dict = result_dict
        self.current_frame = current_frame
        self.stop = False

    def run(self):
        while not self.stop:
            for k in self.result_dict.keys():
                if self.result_dict[k].qsize() > 0:
                    self.current_frame[k] = self.result_dict[k].get()
            time.sleep(0.02)

    def dispose(self):
        self.stop = True