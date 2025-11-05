

import cv2, numpy as np
from PIL import Image

IMG = "image1.tif"  

img_cv  = cv2.imread(IMG)            
img_pil = Image.open(IMG)      

res_cv  = cv2.resize(img_cv, (200, 200))
res_pil = img_pil.resize((200, 200))

left, top, right, bottom = 50, 50, 300, 300
crop_cv  = img_cv[top:bottom, left:right]
crop_pil = img_pil.crop((left, top, right, bottom))

(h, w) = img_cv.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1.0)
rot_cv  = cv2.warpAffine(img_cv, M, (w, h))
rot_pil = img_pil.rotate(45)

zoom_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5)
shrk_cv = cv2.resize(img_cv, None, fx=0.5, fy=0.5)

flip_h = cv2.flip(img_cv, 1)
flip_v = cv2.flip(img_cv, 0)

cv2.imwrite("out_res_cv.png", res_cv)
cv2.imwrite("out_crop_cv.png", crop_cv)
cv2.imwrite("out_rot_cv.png",  rot_cv)
cv2.imwrite("out_zoom_cv.png", zoom_cv)
cv2.imwrite("out_shrk_cv.png", shrk_cv)
cv2.imwrite("out_flip_h.png",  flip_h)
cv2.imwrite("out_flip_v.png",  flip_v)
print("Saved outputs: out_*.png")
