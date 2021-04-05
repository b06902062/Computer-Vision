import numpy as np
def writepoints(filepath, Set):

	flat_list = [[item for sublist in tri_set for item in sublist] for tri_set in Set]
	with open(filepath, "w") as output:
		for line in flat_list:
			string = ' '.join(str(i) for i in line)
			output.write(string + "\n")

def readpoints(filepath):
	ans = []
	with open(filepath, 'r') as f:
		lines = f.readlines()
		for l in lines[::2]:
			point = l.strip().split(',' , 1)
			ans.append(point)
	ans = [[int(i[0]) , int(i[1])] for i in ans]
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
    print(ans)
    return ans

def formula(P, u, v):
	delta = u[0]*v[1] - v[0]*u[1]
	delta_x = P[0]*v[1] - v[0]*P[1]
	delta_y = P[1]*u[0] - u[1]*P[0]
	return [delta_x, delta_y, delta]


def intersect(points, vec):
	#print(points, vec)
	zero = points[0]
	u_vec = points[1] - zero
	v_vec = points[2] - zero
	L0 = formula( vec[0]-zero, u_vec, v_vec )
	L0 = np.array([L0[0]/L0[2], L0[1]/L0[2]])
	L1 = formula( vec[1]-zero, u_vec, v_vec )
	L1 = np.array([L1[0]/L1[2], L1[1]/L1[2]])

	if(((L0[0]+L0[1])<=1 and L0[0]>=0 and L0[1]>=0)
		or ((L1[0]+L1[1])<=1 and L1[0]>=0 and L1[1]>=0)):
		return True
	
	# x=0 intersect
	if(L1[0]- L0[0] != 0): #有解
		alpha = -L0[0]/(L1[0]-L0[0])
		if(0<=alpha<=1 and 0<=(L0[1] + alpha*(L1[1]-L0[1]))<=1 ):
			return True
	elif(L1[1] * L0[1] < 0): #無限多解
		return True

	# y=0 intersect
	if(L1[1]-L0[1] != 0):
		alpha = -L0[1]/(L1[1]-L0[1])
		if(0<=alpha<=1 and 0<=(L0[0] + alpha*(L1[0]-L0[0]))<=1 ):
			return True
	elif(L1[0]*L0[0] < 0):
		return True

	# x+y=1 intersect
	if(L1[0] - L0[0] + L1[1] - L0[1] != 0):

		alpha = (1 - L0[0] - L0[1]) / (L1[0] - L0[0] + L1[1] - L0[1])
		P = L0 + alpha*(L1-L0)
		if(0<=alpha<=1 and (P[0]>=0 and P[1] >= 0)):
			return True
	elif(1 - L0[0] - L0[1] == 0 and (L0[0]*L1[0] < 0)):	#無限多解
		return True
	return False
# v
# ^
# |
# |
# 0 ----> u


def is_cut(points, vec_list):
	for vec in vec_list:
		if(intersect(points,vec)):
			return True
	return False

def TriAngle_main():
	Set_a = readpoints("./data/pointInPic1.txt")
	Set_b = readpoints("./data/pointInPic2.txt")

	Vec_a = readveclist("./data/vectorInPic1.txt")
	Vec_b = readveclist("./data/vectorInPic2.txt")

	Set_a = Set_a + [k[i] for k in Vec_a for i in range(2)]
	Set_b = Set_b + [k[i] for k in Vec_b for i in range(2)]
	
	from scipy.spatial import Delaunay
	import numpy as np

	Arr_a = np.array(Set_a)
	Arr_b = np.array(Set_b)

	tri = Delaunay(Arr_a)

	TriSet_a = Arr_a[tri.simplices]
	TriSet_b = Arr_b[tri.simplices]

	index_tobe_delete = []
	for index in range(len(tri.simplices)):
		if(is_cut(TriSet_a[index], Vec_a) or is_cut(TriSet_b[index], Vec_b)):
			index_tobe_delete.append(index)
	np.delete(TriSet_a, index_tobe_delete)
	np.delete(TriSet_b, index_tobe_delete)
	print(index_tobe_delete)
	writepoints("./data/triangles1.txt", TriSet_a)
	writepoints("./data/triangles2.txt", TriSet_b)

if __name__ == '__main__':
    import os
    os.chdir('..')
    TriAngle_main()