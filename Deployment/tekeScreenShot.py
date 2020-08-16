import cv2

img = cv2.VideoCapture("res/videos/video.mp4")

ret, frame = img.read()

cv2.imwrite("res/images/screenshot.png",frame)