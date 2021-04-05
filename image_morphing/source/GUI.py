# import pyautogui
import cv2 
import numpy as np


draw = 0
refPt = [-1, -1]
matchPoint = []
tmp = []
colorList = [[np.random.randint(255) for i in range(3)] for i in range(1000)]
matchVector = []

def click(event, x, y, flags, image):
    # grab references to the global variables
    global refPt, draw, tmp
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
    
        draw = True
        tmp.append(refPt)
    

def GUI_main(fp1, fp2):


    global draw, tmp, matchPoint
    tmpColor = []
    matchCount = 0
    cv2.namedWindow("eevee")
    
#    print(fp1, os.getcwd())

    img1 = cv2.imread(fp1)

    midLine = img1.copy()[:,:1,:]
    midLine[:,:,:] = [0, 1, 0]
    img2 = cv2.imread(fp2)

    _h, _w = img1.shape[:2]
    side_point = [(0,0),(0,_w-1),(_h-1,0),(_h-1,_w-1)]
    matchPoint += [[y, y] for y in side_point]


    concatImg = (np.concatenate((img1, midLine), axis=1))
    concatImg = np.concatenate((concatImg, img2), axis=1)
    originImg = concatImg.copy()
    cv2.setMouseCallback("eevee", click, concatImg)
    cv2.imshow("eevee", concatImg)

    while True:
        
        
        cv2.imshow("eevee", concatImg)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        if draw:
            color = concatImg[refPt[1]-3:refPt[1]+3, refPt[0]-3:refPt[0]+3, :].copy()
            print(color.shape)
            tmpColor.append(color)
            
            concatImg[refPt[1]-3:refPt[1]+3, refPt[0]-3:refPt[0]+3, :] = colorList[matchCount]
            draw = False
            cv2.imshow("eevee", concatImg)

        if key == ord("v"):

            if len(tmp) != 4:
                print("Error operation, do it again")
                for i,c in enumerate(tmpColor):
                    concatImg[tmp[i][1]-3:tmp[i][1]+3, tmp[i][0]-3:tmp[i][0]+3, :] = tmpColor[i]
                tmp = []
                tmpColor = []
                
                continue

            img1_h, img1_w = img1.shape[:2]
            matchVector.append(([tmp[0],tmp[1]], [[tmp[2][1], tmp[2][0] - img1_w - 1], [tmp[3][1], tmp[3][0] - img1_w - 1]]))
            cv2.line(concatImg, tuple(tmp[0]), tuple(tmp[1]), (0,255,0), 1)
            cv2.line(concatImg, tuple(tmp[2]), tuple(tmp[3]), (0,255,0), 1)
            print("Current Vector: ", matchVector)
            tmp = []
            tmpColor = []

        if key == ord("p"):

            if len(tmp) != 2:
                print("Error operation, do it again")
                for i,c in enumerate(tmpColor):
                    concatImg[tmp[i][1]-3:tmp[i][1]+3, tmp[i][0]-3:tmp[i][0]+3, :] = tmpColor[i]
                tmp = []
                tmpColor = []
                
                continue


            img1_h, img1_w = img1.shape[:2]
            matchPoint.append(([tmp[0][1], tmp[0][0]], [tmp[1][1], tmp[1][0] - img1_w - 1]))
            matchCount += 1
            print("Current Match: ", matchPoint)
            tmp = []
            tmpColor = []

    with open("./data/pointInPic1.txt","w") as f:
        f2 = open("./data/pointInPic2.txt","w")
        for row in matchPoint:

            f.writelines([",".join([str(x) for x in row[0]]) + "\n","===\n"])
            f2.writelines([",".join([str(x) for x in row[1]])+"\n", "===\n"])

    with open("./data/vectorInPic1.txt","w") as f:
        f2 = open("./data/vectorInPic2.txt", "w")
        for row in matchVector:
            f.writelines([",".join([str(x) for x in row[0][0]]) + "\n", ",".join([str(x) for x in row[0][1]]) + "\n", "===\n"])
            f2.writelines([",".join([str(x) for x in row[1][0]]) + "\n", ",".join([str(x) for x in row[1][1]]) + "\n", "===\n"])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import os
    os.chdir('..')
    fp1 = './graph/' + input('graph1: ')
    fp2 = './graph/' + input('graph2: ')
    GUI_main(fp1, fp2)