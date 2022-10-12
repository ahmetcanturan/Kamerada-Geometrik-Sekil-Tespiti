import cv2
import numpy as np
import os

kernel=np.ones((5,5),np.uint8)
dusuk_beyaz = np.array([0, 0,140])
yuksek_beyaz = np.array([256, 60, 256])
kamera = cv2.VideoCapture(0)
while (kamera.isOpened()):

    ret,image=kamera.read()
    img=cv2.flip(image,-1)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # _,mask=cv2.threshold(img_gray,140,255,cv2.THRESH_BINARY)

    mask = cv2.inRange(img_hsv, dusuk_beyaz, yuksek_beyaz)
    blur=cv2.GaussianBlur(mask,(15,15),0)
    _, mask_2 = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
    mask_2=cv2.dilate(mask_2,kernel,iterations=2)
    closing = cv2.morphologyEx(mask_2, cv2.MORPH_CLOSE, kernel)
    erode = cv2.erode(closing, kernel, iterations=2)
    contours = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)  # 0.04 değeri çok önemli
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Position for writing text
        x, y = approx[0][0] # text yazmak için x ,y noktalarını bulur
        if len(approx) == 3:
            cv2.putText(img, "Ucgen", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        elif len(approx) == 4:
            cv2.putText(img, "Dikdortgen", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        elif len(approx) == 5:
            cv2.putText(img, "Besgen", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        else:
            cv2.putText(img, "Daire", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    cv2.imshow("son",img)
    cv2.imshow("th", erode)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()