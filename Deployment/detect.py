from keras.models import load_model
import keras
import cv2
import pickle
import numpy as np
classifier = load_model("res/models/parking_bw_32_2.h5")
video = cv2.VideoCapture(r"res\\videos\\video.mp4")


with open("res/pickles/roi","rb") as fp:
    rois = pickle.load(fp)
rois = rois[1:-1]


while True:
    ret, frame = video.read()
    if ret:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = frame_bw / 255
        for roi in rois:
            parking_spot = frame_bw[roi[0]:roi[1], roi[2]:roi[3]]  #y:y+h,x:x+w
            spot_resized = cv2.resize(parking_spot, (32, 32))
            spot_reshaped = spot_resized.reshape(1,32,32,1)
            predicted = classifier.predict_classes(spot_reshaped)
            status = int(predicted[0][0])
            if(status==0): #spot full
               cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 0, 255), 2)
            else: #spot empty
                cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
