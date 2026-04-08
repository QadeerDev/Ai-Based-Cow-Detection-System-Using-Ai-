import cv2; c=cv2.VideoCapture('input/i4.mp4')
print(c.get(cv2.CAP_PROP_FPS))