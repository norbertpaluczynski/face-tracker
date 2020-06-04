import cv2

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    padding = 15

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20,20)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x-padding, y-2*padding), (x+w+2*padding, y+h+2*padding), (0, 255, 0), 2)
            sub = frame[y-padding:y+w+padding, x-padding:x+h+padding]
            cv2.imshow('window', sub)
            cv2.imshow('window2', sub)
            cv2.imshow('window3', sub)
            cv2.imshow('window4', sub)
            cv2.imshow('window5', sub)
            cv2.imshow('window6', sub)
            cv2.imshow('window7', sub)
            cv2.imshow('window8', sub)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()