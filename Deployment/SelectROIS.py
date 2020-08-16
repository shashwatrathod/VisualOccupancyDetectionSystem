import cv2
import pickle

img = cv2.imread("res/images/screenshot.png")
rois = [] #x,y,w,h
while True:
    r = cv2.selectROI("Select ROIS",img,fromCenter=True)
    rois.append([int(r[1]),int(r[1] + r[3]), int(r[0]),int(r[0] + r[2])]) #y,y+h,x,x+w

    key = cv2.waitKey(0) & 0xFF
    if key == ord("c"):
        break

with open("res/pickles/roi","wb") as fp:
    pickle.dump(rois,fp)

print(rois)