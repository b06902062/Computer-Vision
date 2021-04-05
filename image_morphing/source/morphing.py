import cv2
import numpy as np
import math
import copy

def compute_uv(x, y, tri):
    a = tri[2][0] - tri[0][0]
    b = tri[1][0] - tri[0][0]
    c = x - tri[0][0]
    d = tri[2][1] - tri[0][1]
    e = tri[1][1] - tri[0][1]
    f = y - tri[0][1]
    u = (c*e - f*b)/(a*e - d*b)
    v = (c*d - f*a)/(b*d - e*a)
    return u, v

def result_uv(u, v, tri):
    a = tri[2][0] - tri[0][0]
    b = tri[1][0] - tri[0][0]
    d = tri[2][1] - tri[0][1]
    e = tri[1][1] - tri[0][1]
    x = u*a + v*b + tri[0][0]
    y = u*d + v*e + tri[0][1]
    return x, y

def img_value(img, X, Y):
    x = int(X); y = int(Y)
    if((X==x)&(Y==y)):
        return img[x][y]

    a = X - x
    b = 1 - a
    c = Y - y
    d = 1 - c

    tmp = b*d*img[x][y] + a*d*img[x+1][y] + b*c*img[x][y+1] + a*c*img[x+1][y+1]
    return [int(tmp[0]), int(tmp[1]), int(tmp[2])]

def locate(img, list1, list2, veclist1, veclist2):
    height, width = img.shape[:2]
    image_pad = cv2.copyMakeBorder( img, 0, 1, 0, 1, cv2.BORDER_CONSTANT)

    ans = np.zeros([height, width, 3], dtype = np.uint8)
    colored = np.zeros([height, width], dtype = bool)

    for (tri1, tri2) in zip(list1, list2):
        max_x = max(tri2[0][0], tri2[1][0], tri2[2][0])
        min_x = min(tri2[0][0], tri2[1][0], tri2[2][0])
        max_y = max(tri2[0][1], tri2[1][1], tri2[2][1])
        min_y = min(tri2[0][1], tri2[1][1], tri2[2][1])
        
        pixels = [[i,j] for i in range(min_x, max_x+1, 1) for j in range(min_y, max_y+1, 1)]
        for index in pixels:
            i, j = index
            colored[i][j] = True

            a = tri2[2][0] - tri2[0][0]
            b = tri2[1][0] - tri2[0][0]
            c = i - tri2[0][0]
            d = tri2[2][1] - tri2[0][1]
            e = tri2[1][1] - tri2[0][1]
            f = j - tri2[0][1]
            
            u = (c*e - f*b)/(a*e - d*b)
            v = (c*d - f*a)/(b*d - e*a)

            #u, v = compute_uv(i, j, tri2)
            if(((u>=0) & (v>=0) & ((u+v)<=1))):
                x, y = result_uv(u, v, tri1)
                ans[i][j] = img_value(image_pad, x, y)

    ###vector
    uncolored_point = [[i,j] for i in range(0, height, 1) for j in range(0, width, 1) if colored[i][j] == False]
    for pt in uncolored_point:
        origin_pt = [0, 0]
        WeightSum = 0
        for (vec1, vec2) in zip(veclist1, veclist2):
            u = vec2[1][0] - vec2[0][0]
            v = vec2[1][1] - vec2[0][1]
            u_ = vec1[1][0] - vec1[0][0]
            v_ = vec1[1][1] - vec1[0][1]
            
            l = math.sqrt(u*u + v*v)
            a = (u*(pt[0]-vec2[0][0]) + v*(pt[1]-vec2[0][1]))/(l*l)
            b = (u*(pt[1]-vec2[0][1]) - v*(pt[0]-vec2[0][0]))/(l*l)

            tmp_pt = (vec1[0][0] + a*u_ - b*v_, vec1[0][1] + a*v_ + b*u_)
            weight = (l/(1+b*l))**3 #can change
            
            origin_pt[0] += tmp_pt[0]*weight
            origin_pt[1] += tmp_pt[1]*weight
            WeightSum += weight
        
        origin_pt[0] = origin_pt[0]/WeightSum
        origin_pt[1] = origin_pt[1]/WeightSum
        ans[pt[0]][pt[1]] = img_value(image_pad, origin_pt[0], origin_pt[1])
    return ans

def difference(i, a, list1, list2):
    ans = copy.deepcopy(list1)
    for (tri, tri1, tri2) in zip(ans, list1, list2):
        for (node, node1, node2) in zip(tri, tri1, tri2):
            node[0] = int(((a-i)/a)*node1[0] + (i/a)*node2[0])
            node[1] = int(((a-i)/a)*node1[1] + (i/a)*node2[1])
    return ans

def output_graph(strr, img):
    #cv2.imshow(strr, img); cv2.waitKey(0)
    print(strr+' is saved!')
    cv2.imwrite(('result/' + strr + '.png'), img)

def Normal(x, a2):
    ans = 0.5+0.5*math.erf((x-0.5)/math.sqrt(2*a2))
    return ans

def Morph(img1, img2, list1, list2, gen, var, veclist1, veclist2):
    for i in range(gen):
        tri_tmp = difference(i, gen-1, list1, list2)
        vec_tmp = difference(i, gen-1, veclist1, veclist2)
        
        ans1 = locate(img1, list1, tri_tmp, veclist1, vec_tmp)
        ans2 = locate(img2, list2, tri_tmp, veclist2, vec_tmp)
        
        alpha = Normal(i/(gen-1), var)
        ans = ((1-alpha)*ans1 + alpha*ans2).astype(np.uint8)
        output_graph('generation_'+str(i), ans1)

def readlist(filepath):
    ans = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            tmp = [[0, 0] for _ in range(3)]
            for i in range(3):
                tmp[i] = [int(l[i*2]), int(l[i*2+1])]
            ans.append(tmp)
    return ans

def readveclist(filepath):
    ans = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            l1 = lines[i]
            l2 = lines[i+1] 
            point1 = l1.strip().split(',' , 1)
            point1 = [int(point1[0]), int(point1[1])]
            point2 = l2.strip().split(',' , 1)
            point2 = [int(point2[0]), int(point2[1])]
            ans.append([point1, point2])
    return ans

def Morph_main(fp1, fp2, gen, var):
    trilist1 = readlist('./data/triangles1.txt')
    trilist2 = readlist('./data/triangles2.txt')
    veclist1 = readveclist('./data/vectorInPic1.txt')
    veclist2 = readveclist('./data/vectorInPic2.txt')
    img1 = cv2.imread(fp1)
    img2 = cv2.imread(fp2)

    #yappi.clear_stats()  # clear profiler
    #yappi.set_clock_type('cpu')
    #yappi.start(builtins=True)  # track builtins
    Morph(img1, img2, trilist1, trilist2, gen, var, veclist1, veclist2)
    #yappi.stop()
    #stat = yappi.get_func_stats()
    #stat.save('callgrind.foo.prof', type='callgrind')
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    import os
    os.chdir('..')
    fp1 = './graph/' + input('graph1: ')
    fp2 = './graph/' + input('graph2: ')
    Morph_main(fp1, fp2, 8, 0.01)