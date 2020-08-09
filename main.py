import cv2
import urllib.request
import numpy as np
import time
import argparse

URL = "http://192.168.0.108:8080/video"
URL3 = "http://192.168.0.108:8080/shot.jpg"

parser = argparse.ArgumentParser(description='Choose mode of capturing video')
parser.add_argument('--mode', metavar='mode', type=str, help='Mode of capturing the video', required=True, nargs=1, choices=['webcam', 'ipcam'])
parser.add_argument('--source', metavar='source', type=str, help='Source of the video: URL or webcam ID', nargs=1, default='0')
args = parser.parse_args()
mode = args.mode[0]
source = args.source[0]

class Helper:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        print(self.x, self.y, self.w, self.h)

if __name__ == '__main__':
    if mode == 'webcam':
        cap = cv2.VideoCapture(int(source))
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profileCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    padding = 15

    while True:
        count = 0
        summed = []

        if mode=='webcam':
            _, frame = cap.read()
        elif mode == 'ipcam':
            imgResp = urllib.request.urlopen(source)
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
            frame = cv2.imdecode(imgNp,-1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20,20)
        )

        profiles = profileCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize = (20,20)
        )

        if len(faces) != 0:
            for (x, y, w, h) in faces:
                summed.append(Helper(x, y, w, h))
        
        if len(profiles) != 0:
            for (x, y, w, h) in profiles:
                summed.append(Helper(x, y, w, h))

        for s in summed:
            cv2.rectangle(frame, (s.x-padding, s.y-2*padding), (s.x+s.w+2*padding, s.y+s.h+2*padding), (0, 255, 0), 2)
            count +=1
            sub = frame[s.y-padding:s.y+s.w+padding, s.x-padding:s.x+s.h+padding]

        cv2.putText(frame, f"Detected faces: {str(count)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66,245,105), 2)

        cv2.imshow('ProFaceTracker2000', frame)
        time.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break