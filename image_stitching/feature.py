import cv2
import numpy as np
from numpy import linalg as LA

import plotly.express as px
import math
import rawpy
import imageio
import scipy
from scipy import ndimage
from scipy.ndimage import filters, gaussian_filter
from cv2 import warpAffine
from cv2 import INTER_LINEAR
from scipy.spatial import distance



def G_filter(img):
    height = img.shape[0];  width = img.shape[1]
    G = [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]
    for i in range(5):
        for j in range(5):
            G[i][j] = G[i][j]/273
    ans = np.zeros([height, width], dtype = np.float)
    for i in range(height):
        for j in range(width):
            if((i > 1) & (i < (height-2)) & (j > 1) & (j < (width-2))):
                ans[i][j] = G[0][0]*img[i-2][j-2] + G[0][1]*img[i-2][j-1] + G[0][2]*img[i-2][j] + G[0][3]*img[i-2][j+1] + G[0][4]*img[i-2][j+2] +\
                            G[1][0]*img[i-1][j-2] + G[1][1]*img[i-1][j-1] + G[1][2]*img[i-1][j] + G[1][3]*img[i-1][j+1] + G[1][4]*img[i-1][j+2] +\
                            G[2][0]*img[i][j-2]   + G[2][1]*img[i][j-1]   + G[2][2]*img[i][j]   + G[2][3]*img[i][j+1]   + G[2][4]*img[i][j+2]   +\
                            G[3][0]*img[i+1][j-2] + G[3][1]*img[i+1][j-1] + G[3][2]*img[i+1][j] + G[3][3]*img[i+1][j+1] + G[3][4]*img[i+1][j+2] +\
                            G[4][0]*img[i+2][j-2] + G[4][1]*img[i+2][j-1] + G[4][2]*img[i+2][j] + G[4][3]*img[i+2][j+1] + G[4][4]*img[i+2][j+2]
    return ans

def gradient(img, flag):
    height = img.shape[0];  width = img.shape[1]
    ans = img.copy()
    for i in range(height):
        for j in range(width):
            if(flag):
                if(j < (width-1)):
                    ans[i][j] = img[i][j+1] - img[i][j]
                else:
                    ans[i][j] = img[i][j] - img[i][j-1]
            else:
                if(i < (height-1)):
                    ans[i][j] = img[i+1][j] - img[i][j]
                else:
                    ans[i][j] = img[i][j] - img[i-1][j]
    return ans

def threshold(img, size):
    (height, width) = img.shape[:2]
    ans = np.zeros([height, width], dtype = np.float)
    for i in range(int(height/size)):
        for j in range(int(width/size)):
            lmax = -1; x = 0; y = 0
            for k in range(size):
                for l in range(size):
                    if(img[i*size+k][j*size+l] > lmax):
                        lmax = img[i*size+k][j*size+l]
                        x = k; y = l
            if(lmax > 20000000):
                ans[i*size+x][j*size+y] = lmax
    return ans

def feature(img):
    height = img.shape[0];  width = img.shape[1]
    ans = G_filter(img)
    
    ### compute gradient
    Ix = cv2.Sobel(ans, cv2.CV_16S, 1, 0)
    Iy = cv2.Sobel(ans, cv2.CV_16S, 0, 1)
    #Ix = gradient(ans, 1)
    #Iy = gradient(ans, 0)

    A = Ix.copy(); B = Iy.copy(); C = Ix.copy()
    for i in range(height):
        for j in range(width):
            A[i][j] = Ix[i][j] * Ix[i][j]
            B[i][j] = Iy[i][j] * Iy[i][j]
            C[i][j] = Ix[i][j] * Iy[i][j]
    
    A = G_filter(A); B = G_filter(B); C = G_filter(C)
    lamda = [0, 0]
    for i in range(height):
        for j in range(width):
            tmp = [ [A[i][j], C[i][j]],
                    [C[i][j], B[i][j]]  ]
            lamda = LA.eigvals(tmp)
            ans[i][j] = lamda[0]*lamda[1] - 0.05*(lamda[0]+lamda[1])**2

    return ans

def scale(img, level):
    (height, width) = img.shape[:2]
    h = int(height*level)
    w = int(width*level)
    return cv2.resize(img, (w, h))

def enlarge(img, level):
    (height, width) = img.shape[:2]
    h = int(height*level)
    w = int(width*level)
    ans = np.zeros([h, w], dtype = np.int)
    for i in range(height):
        for j in range(width):
            if((i*level < h) & (j*level < w)):
                ans[i*level][j*level] = img[i][j]
    return ans

def too_close(a, b, rad):
    dis = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    if(dis < rad):
        return True
    else:
        return False

def NMS(img, point_num):
    (height, width) = img.shape[:2]
    dtype = [('x', int), ('y', int), ('r', float)]
    a = np.zeros([10000], dtype = dtype)

    ptr = 0
    for i in range(height):
        for j in range(width):
            if(ptr == 10000):
                break
            if(img[i][j] > 0):
                a[ptr] = (i, j, img[i][j])
                ptr += 1
    
    a = np.resize(a, ptr)
    a[::-1].sort(order = 'r')
    
    tmp = np.zeros([ptr], dtype = dtype); tmp[0] = a[0]
    count = 1; rad = min(height, width)
    while(count < min(point_num, ptr)):
        no_add = True
        for i in range(ptr):
            close = False
            for j in range(count):
                if(too_close(a[i], tmp[j], rad)):
                    close = True
                    break
            if(close):
                continue
            else:
                tmp[count] = a[i]
                count += 1
                no_add = False
        if(no_add):
            rad /= 2
    print(ptr, count, rad)
    
    ans = np.zeros([height, width], dtype = np.float)
    for i in range(count):
        ans[tmp[i][0]][tmp[i][1]] = 255
    return ans

def find_feature_xy(img):
    table = []
    myimg = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (myimg[i][j] == 255):
                table.append([i, j])

    return table

def featureinfo(result, img):
    featuredata = []
    for i in range(len(result)):
        x = result[i][0]
        y = result[i][1]
        featuredata.append([img[x-2][y-2] / 255, img[x-2][y-1] / 255, img[x-2][y] / 255, img[x-2][y+1] / 255, img[x-2][y+2] / 255, 
                            img[x-1][y-2] / 255, img[x-1][y-1] / 255, img[x-1][y] / 255, img[x-1][y+1] / 255, img[x-1][y+2] / 255,
                            img[x][y-2] / 255, img[x][y-1] / 255, img[x][y] / 255, img[x][y+1] / 255, img[x][y+2] / 255, 
                            img[x+1][y-2] / 255, img[x+1][y-1] / 255, img[x+1][y] / 255, img[x+1][y+1] / 255, img[x+1][y+2] / 255,
                            img[x+2][y-2] / 255, img[x+2][y-1] / 255, img[x+2][y] / 255, img[x+2][y+1] / 255, img[x+2][y+2] / 255 ])

    return featuredata

def matching(featuredata, featuredata2, result, result2):
    pair = []
    for i,g in enumerate(featuredata):
        minsum = 100
        secminsum = 100
        tmp = -1
        for j,f in enumerate(featuredata2):
            total = 0
            for k in range(25):
                total += (f[k]-g[k]) ** 2
            if (total <= minsum):
                secminsum = minsum
                minsum = total
                tmp = j

        minsum2 = 100
        secminsum2 = 100
        tmp2 = -1
        for l, m in enumerate(featuredata):
            total2 = 0
            for n in range(25):
                total2 += (featuredata2[tmp][n] - m[n]) ** 2
            if (total2 <= minsum):
                secminsum2 = minsum2
                minsum2 = total2
                tmp2 = l

        if (tmp2 == i):
            if (minsum <= secminsum*0.4) and (minsum2 <= secminsum2*0.4):
                if (np.abs(result2[tmp][0] - result[i][0]) <= 15 ) and (np.abs(result2[tmp][1] - result[i][1]) <= 200):
                    pair.append([result[i][0], result[i][1], result2[tmp][0], result2[tmp][1]])

    return pair

###read file

img = cv2.imread('_DSC1799.JPG', 0)
img = scale(img, 0.1)
ans = img.copy()

for j in range(4):
    tmp = scale(img, 0.5**j)
    tmp_ans = feature(tmp)
    tmp_ans = enlarge(tmp_ans, 2**j)
    ans = np.where(tmp_ans > ans, tmp_ans, ans)

ans = threshold(ans, 1)
ans = NMS(ans, 100)


result = find_feature_xy(ans)

featuredata = featureinfo(result, img)

cv2.imwrite('ans.jpg', ans)

for i in range(len(result)):
    x = result[i][0]
    y = result[i][1]
    cv2.circle(img, (y, x), 2, (255, 0, 0), -1)

cv2.imwrite('look.jpg', img)    


print('############################')

img2 = cv2.imread('_DSC1800.JPG', 0)
img2 = scale(img2, 0.1)
ans2 = img2.copy()
print(img2.shape[0], img2.shape[1])

for j in range(4):
    tmp2 = scale(img2, 0.5**j)
    tmp_ans2 = feature(tmp2)
    tmp_ans2 = enlarge(tmp_ans2, 2**j)
    ans2 = np.where(tmp_ans2 > ans2, tmp_ans2, ans2)

ans2 = threshold(ans2, 1)
ans2 = NMS(ans2, 100)


result2 = find_feature_xy(ans2)

featuredata2 = featureinfo(result2, img2)

cv2.imwrite('ans2.jpg', ans2)

for i in range(len(result2)):
    x = result2[i][0]
    y = result2[i][1]
    cv2.circle(img2,(y, x), 2, (255, 0, 0), -1)

cv2.imwrite('look2.jpg', img2)

print('############################')

pair = matching(featuredata, featuredata2, result, result2)
print(pair)

for i in range(len(pair)):
    x1 = pair[i][0]
    y1 = pair[i][1]
    x2 = pair[i][2]
    y2 = pair[i][3]
    cv2.circle(ans, (y1, x1), 3, (255, 255, 0), 1)
    cv2.circle(ans2, (y2, x2), 3, (255, 255, 0), 1)

cv2.imwrite('find.jpg', ans)

cv2.imwrite('find2.jpg', ans2)


print('############################')

stack = np.hstack((ans,ans2))

for i in range(len(pair)):
    cv2.line(stack, (pair[i][1], pair[i][0]), (pair[i][3]+400, pair[i][2]), (255, 0, 0), 1)

cv2.imwrite('stack.jpg', stack)

for i in range(len(pair)):
    x1 = pair[i][0]
    y1 = pair[i][1]
    x2 = pair[i][2]
    y2 = pair[i][3]
    cv2.circle(img, (y1, x1), 3, (0, 0, 255), 1)
    cv2.circle(img2, (y2, x2), 3, (0, 0, 255), 1)

stack2 = np.hstack((img,img2))

for i in range(len(pair)):
    cv2.line(stack2, (pair[i][1], pair[i][0]), (pair[i][3]+400, pair[i][2]), (255, 255, 0), 1)


cv2.imwrite('stack2.jpg', stack2)



