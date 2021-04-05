import numpy as np
import cv2
import math

def scale(img, level):
    (height, width) = img.shape[:2]
    h = int(height*level)
    w = int(width*level)
    return cv2.resize(img, (w, h))

def Cylindrical_proj(img, f):
    (height, width) = img.shape[:2]
    mx = width/2; my = height/2

    ans = img.copy()
    for i in range(height):
        for j in range(width):
            x = f*math.tan((j-mx)/f)
            y = (i - my)*(x**2+f**2)**0.5/f
            x = int(x + mx); y = int(y + my)
            if((x < 0) | (x >= width) | (y < 0) | (y >= height)):
                ans[i][j] = 0
            else:
                ans[i][j] = img[y][x]
    
    return ans       

def splice(proj_final, proj_final2, final_dx, final_dy):
    proj_final = np.array(proj_final)
    proj_final2 = np.array(proj_final2)

    new_x = proj_final.shape[0] + proj_final2.shape[0] - (proj_final.shape[0] - final_dx)
    new_y = proj_final.shape[1] + proj_final2.shape[1] - (proj_final.shape[1] - final_dy)

    final_result = np.zeros((new_x, new_y, 3))

    for i in range(proj_final.shape[0]):
        for j in range(proj_final.shape[1]):
            final_result[i][j] = proj_final[i][j]

    for i in range(proj_final2.shape[0]):
        for j in range(proj_final2.shape[1]):
            if (i < (proj_final.shape[0] - final_dx)) and (j < proj_final.shape[1] - final_dy):
                final_result[i + final_dx][j + final_dy] = (final_result[i + final_dx][j + final_dy]*(1 - j/(proj_final.shape[1] - final_dy)) + 
                                                            proj_final2[i][j] * (j/(proj_final.shape[1] - final_dy)))
            else:
                final_result[i + final_dx][j + final_dy] = proj_final2[i][j]

    return final_result

def projection(img, f, scalenum):
    img = scale(img, scalenum)
    proj = Cylindrical_proj(img, f)
    dx = int((proj.shape[1]/2) + f*math.atan(-(proj.shape[1]/2)/f))+1
    proj_new = proj[::, dx:]
    proj_final = np.delete(proj_new, [386, 387, 388, 389, 390, 391, 392], 1)

    return proj_final

def find_dx_dy(pair, scalenum):
    disx = []
    disy = []
    for i in range(len(pair)):
        dx = pair[i][0] - pair[i][2]
        dy = pair[i][1] - pair[i][3]
        disx.append(dx)
        disy.append(dy)

    disx = np.sort(disx)
    middlex = int(len(disx)/2) - 1
    new_dx = []
    sumx = 0
    for i in range(len(disx)):
        if (disx[i] >= 0) and (disx[i] <= disx[middlex]*2) and (disx[i] >= disx[middlex]/2):
            new_dx.append(disx[i])
            sumx += disx[i]
    final_dx = int(sumx/len(new_dx))
       

    disy = np.sort(disy)
    middley = int(len(disy)/2) - 1 
    new_dy = []
    sumy = 0

    for j in range(len(disy)):
        if (disy[j] >= 0) and (disy[j] <= disy[middley]*1.2) and (disy[j] >= disy[middley]/(1.2)):
            new_dy.append(disy[j])
            sumy += disy[j]

    final_dy = int(sumy/len(new_dy))

    final_dx = int(final_dx * (scalenum * 10))
    final_dy = int(final_dy * (scalenum * 10))

    return final_dx, final_dy


f = 644
scalenum = 0.1
img1799 = cv2.imread('_DSC1799.jpg')
proj1799_final = projection(img1799, f, scalenum)
#cv2.imwrite('proj1799.jpg', proj1799_final)

img1800 = cv2.imread('_DSC1800.jpg')
proj1800_final = projection(img1800, f, scalenum)
#cv2.imwrite('proj1800.jpg', proj1800_final)

img1801 = cv2.imread('_DSC1801.jpg')
proj1801_final = projection(img1801, f, scalenum)
#cv2.imwrite('proj1801.jpg', proj1801_final)

img1802 = cv2.imread('_DSC1802.jpg')
proj1802_final = projection(img1802, f, scalenum)
#cv2.imwrite('proj1802.jpg', proj1802_final)

img1803 = cv2.imread('_DSC1803.jpg')
proj1803_final = projection(img1803, f, scalenum)
#cv2.imwrite('proj1803.jpg', proj1803_final)

img1804 = cv2.imread('_DSC1804.jpg')
proj1804_final = projection(img1804, f, scalenum)
#cv2.imwrite('proj1804.jpg', proj1804_final)

total_pair = []

pair1803_1804 = [[104, 172, 96, 52], [120, 272, 116, 156], [290, 274, 287, 152], [320, 204, 318, 83], [330, 264, 327, 143], [364, 205, 363, 85], [380, 264, 376, 144], [389, 190, 389, 70], [399, 214, 398, 95], [412, 177, 421, 113]]
pair1802_1803 = [[96, 328, 104, 172], [193, 151, 192, 50], [200, 340, 200, 208], [212, 308, 208, 144], [240, 309, 238, 145], [283, 218, 281, 53], [304, 184, 304, 18], [313, 211, 312, 47], [315, 177, 315, 11], [340, 333, 335, 168], [382, 321, 376, 158], [411, 381, 399, 214], [417, 326, 412, 177], [424, 118, 416, 12]]
pair1801_1802 = [[152, 269, 150, 134], [179, 254, 183, 245], [180, 100, 179, 184], [180, 304, 180, 168], [195, 286, 193, 151], [262, 165, 259, 24], [302, 190, 301, 52], [319, 313, 315, 177], [356, 377, 349, 237], [432, 192, 432, 72], [436, 244, 448, 80], [466, 167, 456, 50]]
pair1800_1801 = [[102, 216, 94, 72], [132, 270, 128, 128], [155, 245, 150, 103], [255, 297, 252, 153], [451, 372, 436, 229]]
pair1799_1800 = [[20, 320, 21, 165], [172, 264, 168, 112], [215, 198, 208, 40], [311, 281, 305, 122], [361, 304, 353, 146]]
total_pair.append(pair1803_1804)
total_pair.append(pair1802_1803)
total_pair.append(pair1801_1802)
total_pair.append(pair1800_1801)
total_pair.append(pair1799_1800)


dx1803_1804, dy1803_1804 = find_dx_dy(total_pair[0], scalenum)
result1803_1804 = splice(proj1803_final, proj1804_final, dx1803_1804, dy1803_1804)
#cv2.imwrite('result1803_1804.jpg', result1803_1804)

dx1802_1803, dy1802_1803 = find_dx_dy(total_pair[1], scalenum)
result1802_1803 = splice(proj1802_final, result1803_1804, dx1802_1803, dy1802_1803)
#cv2.imwrite('result1802_1803.jpg', result1802_1803)

dx1801_1802, dy1801_1802 = find_dx_dy(total_pair[2], scalenum)
result1801_1802 = splice(proj1801_final, result1802_1803, dx1801_1802, dy1801_1802)
#cv2.imwrite('result1801_1802.jpg', result1801_1802)

dx1800_1801, dy1800_1801 = find_dx_dy(total_pair[3], scalenum)
result1800_1801 = splice(proj1800_final, result1801_1802, dx1800_1801, dy1800_1801)
#cv2.imwrite('result1800_1801.jpg', result1800_1801)

dx1799_1800, dy1799_1800 = find_dx_dy(total_pair[4], scalenum)
result1799_1800 = splice(proj1799_final, result1800_1801, dx1799_1800, dy1799_1800)
cv2.imwrite('result1799_1800.jpg', result1799_1800)
