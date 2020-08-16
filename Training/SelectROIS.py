import cv2
import pickle

img = cv2.imread("8.jpg")
i=47054
try:
    with open("i","rb") as f:
        i = int(pickle.load(f))
except:
    pass
while True:
    r = cv2.selectROI("Select ROIS",img,fromCenter=True)
    cv2.imwrite("dataset/train/spot/spot"+str(i)+".jpg",img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]) #y,y+h,x,x+w
    i += 1

    key = cv2.waitKey(0) & 0xFF
    if key == ord("c"):
        break

with open("i","rb") as f:
    pickle.dump(i,f)