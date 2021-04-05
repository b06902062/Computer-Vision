import cv2
import numpy as np
import math
import rawpy
import imageio
import os
#import plotly.express as px

###path set up
path = "./tiff"
if not os.path.isdir(path):
    os.mkdir(path)

file = ['./tiff/1.tiff', './tiff/2.tiff', './tiff/3.tiff', './tiff/4.tiff', './tiff/5.tiff']
###

###nef to tiff
for i in range(5):
    rawpath='./raw/' + str(i+1) + '.nef'
    with rawpy.imread(rawpath) as raw:
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    imageio.imsave(file[i], rgb)
###

###read file
img = [None for i in range(5)]
for i in range(5):
    img[i] = cv2.imread(file[i], -1)
height = img[0].shape[0]; width = img[0].shape[1]
###

###HDR
t = [0.5, 0.25, 0.125, 1/15, 1/30]
h_map = np.zeros([height, width], dtype=np.int)
r_map = np.zeros([height, width, 3], dtype=np.int)
L_sum = 0; theta = 0.0001

for i in range(height):
    for j in range(width):
        R = 0; G = 0; B = 0; p = 5
        for k in range(p):
            R += img[k][i, j][0]/t[k]
            G += img[k][i, j][1]/t[k]
            B += img[k][i, j][2]/t[k]
        r_map[i, j, 0] = R/p
        r_map[i, j, 1] = G/p
        r_map[i, j, 2] = B/p
        h_map[i, j] = (R/3+G/3+B/3)/p
    if ((i%100)==0): print(i)
#heat = px.imshow(h_map); heat.show()
cv2.imwrite('hdr.hdr', r_map)
###

###tone mapping
r_map = cv2.imread('hdr.hdr', -1)
h_map = np.zeros([r_map.shape[0], r_map.shape[1]], dtype=np.int)
L_sum = 0; theta = 0.0001

for i in range(r_map.shape[0]):
    for j in range(r_map.shape[1]):
        h_map[i, j] = 0.27*r_map[i, j, 0] + 0.67*r_map[i, j, 1] + 0.06*r_map[i, j, 2]
        L_sum += math.log(theta + h_map[i, j])

tone = np.zeros(r_map.shape, np.uint8)
L_mean = math.exp(L_sum/(r_map.shape[0]*r_map.shape[1]))
alpha = 0.18; L_white = 7.5

for i in range(r_map.shape[0]):
    for j in range(r_map.shape[1]):
        L_m = alpha*h_map[i, j]/L_mean
        L_d = L_m*(1 + L_m/(L_white*L_white))/(1 + L_m)
        if L_d > 1: L_d = 1
        R = min(255, 255*L_d*(r_map[i, j, 0]/h_map[i, j]))
        G = min(255, 255*L_d*(r_map[i, j, 1]/h_map[i, j]))
        B = min(255, 255*L_d*(r_map[i, j, 2]/h_map[i, j]))
        tone[i, j, 0] = R; tone[i, j, 1] = G; tone[i, j, 2] = B
    if ((i%100)==0): print(i)

cv2.imwrite('tone.jpg', tone)
cv2.imshow('tone', tone)
cv2.waitKey(0)
cv2.destroyAllWindows()
###
