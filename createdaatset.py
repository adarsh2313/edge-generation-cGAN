import cv2
import os
import numpy as np
import random
random.seed(234)

img = []
for _ in range(1000):
    img.append(np.random.randint(0,128,(8,8,1),np.uint8))

# For first 500 images, horizontal edges:

for i in range(500):
    s = random.randint(2,6)
    img[i] = cv2.rectangle(img[i],(0,s),(8,8),(255),-1)
    cv2.imwrite('/Users/Adarsh/Desktop/projects/cGAN-edge-generation/Dataset/horizontal/image{:03d}.png'.format(i+1),img[i])

# For next 500 images, vertical edges:

for i in range(500,1000):
    s = random.randint(2,6)
    img[i] = cv2.rectangle(img[i],(s,0),(8,8),(255),-1)
    cv2.imwrite('/Users/Adarsh/Desktop/projects/cGAN-edge-generation/Dataset/vertical/image{:03d}.png'.format(i+1),img[i])