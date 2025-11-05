# prog6_morphology.py
# pip install opencv-python matplotlib numpy

import cv2, numpy as np, matplotlib.pyplot as plt
IMG = "images/image6-1.png"
img = cv2.imread(IMG, 0)  # binary or grayscale

kernel = np.ones((5,5), np.uint8)
erosion  = cv2.erode(img,  kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel)
closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(10,8))
pairs = [("Original",img),("Erosion",erosion),("Dilation",dilation),
         ("Opening",opening),("Closing",closing),("Gradient",gradient)]
for i,(t,im) in enumerate(pairs,1):
    plt.subplot(2,3,i); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
plt.tight_layout(); plt.show()

# try different structuring elements
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
d_cross = cv2.dilate(img, cross, 1)
cv2.imwrite("out_dilate_cross.png", d_cross)
