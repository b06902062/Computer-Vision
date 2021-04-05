import cv2
import numpy as np
import math
import copy

def locate(veclist1, veclist2): 
    height, width = img.shape[:2]
    image_pad = cv2.copyMakeBorder( img, 0, 1, 0, 1, cv2.BORDER_CONSTANT)

    ans = np.zeros([height, width, 3], dtype = np.uint8)
    
    #...
    
    for pt in uncolored_point:
        origin_pt = [0, 0]
        WeightSum = 0
        for (vec1, vec2) in zip(veclist1, veclist2):
            u = vec2[1][0] - vec2[0][0]
            v = vec2[1][1] - vec2[0][1]
            u_ = vec1[1][0] - vec1[0][0]
            v_ = vec1[1][1] - vec1[0][1]
            
            l = math.sqrt(u*u + v*v)
            a = (u*pt[0] + v*pt[1])/(l*l)
            b = (u*pt[1] - v*pt[0])/(l*l)

            tmp_pt = (vec1[0][0] + a*u_ - b*v_, vec1[0][1] + a*v_ + b*u_)
            weight = (l/(1+b*l))**3 #can change
            
            origin_pt[0] += tmp_pt[0]*weight
            origin_pt[1] += tmp_pt[1]*weight
            WeightSum += weight
        
        origin_pt[0] = origin_pt[0]/WeightSum
        origin_pt[1] = origin_pt[1]/WeightSum
        ans[pt[0]][pt[1]] = img_value(img, origin_pt)
    return ans