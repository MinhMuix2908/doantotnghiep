from flask import Flask, Response, url_for, render_template
from flask import request
from flask_cors import CORS, cross_origin
import threading
import cv2

class WebService(threading.Thread):
    def __init__(self, frame_dict, cfg):
        threading.Thread.__init__(self)
        self.app = Flask(__name__)
        cors = CORS(self.app)
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.frame_dict = frame_dict
        self.cfg = cfg

        @self.app.route('/')
        def index():
            """Video streaming home page."""
            return render_template("index.html")
        
        @self.app.route('/get_stream/<int:deviceID>', methods=['GET', 'POST'])
        @cross_origin()
        def get_stream(deviceID):
            # deviceID = request.form['deviceID']
            return Response(gen_frame(deviceID), mimetype='multipart/x-mixed-replace; boundary=frame')            
        
        def gen_frame(deviceID):
            while True:
                if deviceID not in self.frame_dict.keys():
                    # frame = cv2.imread('no-signal.jpg')
                    continue
                else:
                    frame = self.frame_dict[deviceID]
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
    def run(self):
        self.app.run(host=self.cfg.web_service.ip, port=self.cfg.web_service.port, threaded=True)
