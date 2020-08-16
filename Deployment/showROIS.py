import cv2
import pickle

regions = []
with open("res/pickles/roi","rb") as fp:
    regions = pickle.load(fp)

image = cv2.imread("res/images/screenshot.png")
print(regions)

for region in regions:
    cv2.rectangle(image,(region[2],region[0]),(region[3],region[1]),(0,255,0),2)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()