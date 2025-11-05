# prog3_smooth_sharpen.py
# pip install opencv-python matplotlib numpy

import cv2, numpy as np, matplotlib.pyplot as plt
IMG = "image3.tif"
img = cv2.imread(IMG)

avg   = cv2.blur(img, (5,5))
gauss = cv2.GaussianBlur(img, (5,5), 1.0)
sharp = cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))

plt.figure(figsize=(10,8))
titles = ["Original","Averaging","Gaussian","Sharpened"]
images = [img, avg, gauss, sharp]
for i,(t,im) in enumerate(zip(titles,images),1):
    plt.subplot(2,2,i); plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.title(t); plt.axis('off')
plt.tight_layout(); plt.show()
