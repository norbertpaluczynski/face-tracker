import cv2
import urllib.request
import numpy as np
import time
import argparse
import requests
import json
import base64
import sqlite3
import uuid
import copy
import face_recognition
import pathlib
import datetime
import subprocess
import sys
from os import listdir
from os.path import isfile, join
from shutil import copyfile


parser = argparse.ArgumentParser(description='Choose mode of capturing video')
parser.add_argument('--mode', metavar='mode', type=str, help='Mode of capturing the video', required=True, nargs=1, choices=['webcam', 'ipcam'])
parser.add_argument('--source', metavar='source', type=str, help='Source of the video: URL or webcam ID', nargs=1, default='0')
parser.add_argument('--format', metavar='format', type=str, help='Input format', required=False, nargs=1, default='0', choices=['image', 'video'])
args = parser.parse_args()
mode = args.mode[0]
source = args.source[0]
input_format = args.format[0]

connection = sqlite3.connect('database.db')
conn = connection.cursor()

#creating table for storing info about recognized people
conn.execute('''CREATE TABLE IF NOT EXISTS people (
    person_id TEXT NOT NULL,
    name TEXT DEFAULT 'UNKNOWN' NOT NULL
)''')

#creating table for storing assignments of faces to people
conn.execute('''CREATE TABLE IF NOT EXISTS faces (
    face_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    date TEXT NOT NULL
)''')

connection.commit()


#creating required directories
pathlib.Path("unique_faces").mkdir(parents=True, exist_ok=True)
pathlib.Path("all_faces").mkdir(parents=True, exist_ok=True)
pathlib.Path("all_images").mkdir(parents=True, exist_ok=True)
pathlib.Path("temp").mkdir(parents=True, exist_ok=True)


def add_person_to_database(face_image_path, face_uuid, image_uuid, datetime):
    person_uuid = uuid.uuid4()
    conn.execute(f"INSERT INTO people VALUES ('{person_uuid}', 'UNKNOWN')")
    conn.execute(f"INSERT INTO faces VALUES ('{face_uuid}', '{person_uuid}', '{image_uuid}', '{datetime}')")
    connection.commit()
    copyfile(face_image_path, f"unique_faces\\{face_uuid}.jpg")

    return (face_uuid, face_uuid)

def add_face_to_database(result_uuid, face_uuid, image_uuid, datetime):
    conn.execute('''
                SELECT person_id 
                FROM faces
                WHERE face_id=?''', (f'{result_uuid}',))
    person_uuid = conn.fetchone()
    conn.execute(f"INSERT INTO faces VALUES ('{face_uuid}', '{person_uuid[0]}', '{image_uuid}', '{datetime}')")
    connection.commit()

    return (result_uuid, face_uuid)


if __name__ == '__main__':
    if mode == 'webcam':
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    padding = 15

    while True:
        if mode=='webcam' or (mode == 'ipcam' and input_format=='video'):
            _, frame = cap.read()
        elif mode == 'ipcam' and input_format == 'image':
            imgResp = urllib.request.urlopen(source)
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
            frame = cv2.imdecode(imgNp, -1)

        original_frame = copy.copy(frame)    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        retval, buffer = cv2.imencode('.jpg', frame)
        jpgsend = base64.b64encode(buffer)
        
        frontfaces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20,20)
        )
        
        for (x, y, w, h) in frontfaces:
            cv2.rectangle(frame, (x - padding, y - 2*padding), (x + w + 2*padding, y + h + 2*padding), (0, 255, 0), 2)

        cv2.putText(frame, f"Detected faces: {str(len(frontfaces))}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66,245,105), 2)
        cv2.imshow('ProFaceTracker2000', frame)

        
        #'space' press event
        if cv2.waitKey(1) & 0xFF == ord(' '):

            #get actual datetime
            frame_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


            #generate id for actual processing frame and save it
            frame_uuid = uuid.uuid4()
            cv2.imwrite(f"all_images\\{frame_uuid}.jpg", original_frame)
            cv2.imwrite("temp\\image.jpg", original_frame)


            #cutting all faces from actual processing frame and asigning uuid for them            
            cut_faces = [] #<=== list of tuple(img, uuid)
            for (x, y, w, h) in frontfaces:
                cut_faces.append((original_frame[y:y+h, x:x+w], uuid.uuid4(), [x, y, w, h]))


            #save cutted faces 
            cut_faces_file_paths = [] #<=== list of tuple(file path, uuid)
            for cf in cut_faces:
                path = f"all_faces\\{cf[1]}.jpg"
                cv2.imwrite(path, cf[0])
                cut_faces_file_paths.append((path, cf[1]))


            matches = [] #<=== list of tuple(uuid z bazy, uuid wyciętej twarzy)
            for cffp in cut_faces_file_paths:
                cmd = ["PowerShell", "-ExecutionPolicy", "Unrestricted", "-Command", f"face_recognition .\\unique_faces {cffp[0]} --cpus -1"]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
                result = p.stdout.read().strip().split(',')[-1]
                if result == 'unknown_person':
                    x = add_person_to_database(cffp[0], cffp[1], frame_uuid, frame_datetime)
                    matches.append(x)
                elif result == 'no_persons_found':
                    continue
                else:                    
                    x = add_face_to_database(result, cffp[1], frame_uuid, frame_datetime)
                    matches.append(x)

            conn = connection.cursor()

            recognition_results = []
            for match in matches:
                conn.execute('''
                SELECT f.face_id, f.person_id, p.name 
                FROM faces f
                INNER JOIN people p ON f.person_id = p.person_id 
                WHERE face_id=?''', (f'{match[1]}',))
                result = conn.fetchone()
                recognition_results.append(result)

            print("\nresults:")
            for x in recognition_results:
                print(x)
                for y in cut_faces:
                    if x[0] == str(y[1]):
                        # x = y[2][0] left
                        # y = y[2][1] top
                        # w = y[2][2] right 
                        # h = y[2][3] bottom
                        cv2.rectangle(original_frame, (y[2][0], y[2][1]), (y[2][0] + y[2][2], y[2][1] + y[2][3]), (0, 0, 255), 2)
                        cv2.rectangle(original_frame, (y[2][0], y[2][1] + y[2][3]), (y[2][0] + y[2][2], y[2][1] + y[2][3] + 25), (0, 0, 255), cv2.FILLED)
                        cv2.putText(original_frame, x[2], (y[2][0] + 6, y[2][1] + y[2][3] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('detected', original_frame)            


        #'q' press event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()