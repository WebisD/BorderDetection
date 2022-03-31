import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

#Importa e converta para RGB
img = cv2.imread('FEI01.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur
img_blur = cv2.blur(img,(7,7))

#Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, 127, a,cv2.THRESH_BINARY_INV)


kernel = np.ones((7,7), np.uint8)

# Er Di
img_dilate = cv2.dilate(thresh,kernel,iterations = 1)
img_close = cv2.morphologyEx(img_dilate, cv2.MORPH_CLOSE, kernel)
#img_erode = cv2.erode(img_dilate,kernel,iterations = 1)


edges_blur = cv2.Canny(image=img_close, threshold1=a/2, threshold2=a/2)

# contorno
contours, hierarchy = cv2.findContours(
                                   image = edges_blur,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)

imagens = [img, thresh, img_dilate, img_close, edges_blur, final]
#imagens = [img,img_blur,img_gray,edges_gray,edges_blur,thresh,thresh_open,final]
formatoX = math.ceil(len(imagens)**.5)
if (formatoX**2-len(imagens))>formatoX:
    formatoY = formatoX-1
else:
    formatoY = formatoX

for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i],'gray')
    plt.xticks([]),plt.yticks([])
plt.savefig("process.png")
plt.show()

plt.imshow(final)
plt.savefig("final.png")