# Module: outlierDetector
# Defines methods to detect outliers in a given set of trajectories
# References: 
# Trajectory Outlier Detection: A Partition-and-Detect Framework, Lee, Han, Li
# Trajectory Clustering: A Partition-and-Group Framework, Jae-Gil Lee, Jiawei Han, Kyu-Young Whang
# -------------------------------------------------------------------------------------------------

import numpy as np, math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

# -------------------------------------------------------------------------------------------------
# DEFINITIONS

# CTR_count: count of the trajectories close to the t-partition Li
# D: parameter given by user, the threshold for closeness of two t-partitions
# F: parameter given by user, the threshold for marking a trajectory as outlier
# p: parameter given by user, the factor used in marking a t-partition as outlier 
# density: The count of t-partitions that are within sd distance to the given t-partition

# -------------------------------------------------------------------------------------------------
# METHODS

# This method returns the standard deviation of the elements in a list
def standard_deviation(l):
	return np.std(np.array(l))


# Distance between two t-partitions
# This is not the distance measure used in the paper by Lee et. al.
# def tp_distance(L1, L2):
# 	# print("L1 = ", L1, "L2 = ", L2)
# 	s1x, s1y, e1x, e1y = L1[0][0], L1[0][1], L1[1][0], L1[1][1]
# 	s2x, s2y, e2x, e2y = L2[0][0], L2[0][1], L2[1][0], L2[1][1]
# 	D = np.linalg.norm([s2y-s1y, s2x-s1x]) + np.linalg.norm([e2y-e1y, e2x-e1x])
# 	return D

# This function returns the projection point P2 of a point P1 on a line L
def projPointOnLine(point, line_2pt):
    # The point is of the form x,y and line_pt contains p1x,p1y,p2x,p2y
    line = [1,1,1,1]

    line[0]= line_2pt[0]
    line[1]= line_2pt[1];
    line[2]= line_2pt[2]-line_2pt[0];
    line[3]= line_2pt[3]-line_2pt[1];
        
    # direction vector of the line
    vx = line[2];
    vy = line[3];

    # difference of point with line origin
    dx = point[0] - line[0];
    dy = point[1] - line[1];
    #print(line_2pt)
    # Position of projection on line, using dot product
    tp = (dx* vx + dy* vy ) / (vx * vx + vy * vy);

    # convert position on line to cartesian coordinates
    point[0] = line[0] + tp* vx;
    point[1] = line[1] + tp* vy;
    return [point[0],point[1]]

#This method determines the all the three distances between L1 and L2  
def tp_distance(L1, L2):

    w1, w2, w3 = 1.0, 1.0, 1.0

    if length(L1) > length(L2):
        temp = deepcopy(L2)
        L2 = deepcopy(L1)
        L1 = deepcopy(temp)

    point1 = [L2[0][0], L2[0][1]]
    line_2pt1 = [L1[0][0], L1[0][1], L1[1][0], L1[1][1]] 
    proj1 = projPointOnLine(point1, line_2pt1)

    point2 = [L1[0][0], L1[0][1]]
    line_2pt2 = [L2[0][0], L2[0][1], L2[1][0], L2[1][1]] 
    proj2 = projPointOnLine(point2, line_2pt2)
    
    six, siy, eix, eiy = L1[0][0], L1[0][1], L1[1][0], L1[1][1]
    sjx, sjy, ejx, ejy = L2[0][0], L2[0][1], L2[1][0], L2[1][1]

    #Computing the perpendicular distance
    lper1 = np.linalg.norm([sjy-proj1[1], sjx-proj1[0]])
    lper2 = np.linalg.norm([siy-proj2[1], six-proj2[0]])

    Dper = (math.pow(lper1,2)+ math.pow(lper2,2))/(lper1+lper2)
    
    #Computing the parallel distance
    lpar1 = min(np.linalg.norm([siy-proj1[1], six-proj1[0]]),np.linalg.norm([eiy-proj1[1], eix-proj1[0]]))
    lpar2 = min(np.linalg.norm([siy-proj2[1], six-proj2[0]]),np.linalg.norm([eiy-proj2[1], eix-proj1[0]]))
    Dpar = min(lpar1, lpar2)
    
    #Computing the angle distance
    x1 = ejx-sjx
    y1 = ejy-sjy
    x2 = eix-six
    y2 = eiy-siy
    inner_product = x1*x2 + y1*y2
    costheta = inner_product/(np.linalg.norm([x1, y1])*np.linalg.norm([x2, y2]))
    #print(x1, y1, x2, y2, inner_product, (np.linalg.norm([x1, y1])*np.linalg.norm([x2, y2])), costheta)
    if costheta < -1:
    	costheta = -1
    elif costheta > 1:
    	costheta = 1
    angle = math.acos(costheta)

    if angle < (math.pi/2):
        Dang = math.sin(angle)*length(L2)
    else:
        Dang = length(L2) 

    return w1*Dper + w2*Dpar + w3*Dang
 

# This method returns the length of a t-partition
def length(segment):
	sx, sy, ex, ey = segment[0][0], segment[0][1], segment[1][0], segment[1][1]
	return np.linalg.norm([ex-sx, ey-sy])

# Minimum Description Length Cost with partitioning
def mdl_par(t, s, e):
    pfactor = 2
    # This is the mdl cost when we partition the trajectory and do not keep the original points
    ld = length([t[s],t[e]])
    # Calculate ldh here
    x1, y1, x2, y2 = t[s][0], t[s][1], t[e][0], t[e][1]
    Dx, Dy = x2-x1, y2-y1
    D = np.linalg.norm([Dx, Dy])
    d = 0
    for i in range(s, e):
        x0, y0 = t[i][0], t[i][1]
        di = math.fabs((Dy*x0 - Dx*y0 + x2*y1 - y2*x1)/D)
        d = d + di
    ldh = d/pfactor
    mdl = ld+ldh
    return mdl

# Minimum Description Length Cost without partitioning
def mdl_nopar(t, s, e):
    tlen = 0
    for i in range(s,e):
        tlen = tlen + length([t[i],t[i+1]])
    return tlen

# Trajectory partitioning algorithm
def mdl_partition(t):
    cp = [t[0]]
    si, l = 1, 1
    while si + l <= len(t):
        ci = si + l
        cost_par = mdl_par(t,si,ci)
        cost_nopar = mdl_nopar(t, si,ci)
        # If partitioning cost is greater than the no-partitioning cost, keep the original point
        if cost_par > cost_nopar:
            cp.append(t[ci-1])
            si, l = ci-1, 1
        else:
            l = l + 1
    cp.append(t[-1])


def partition(T):
	L = []
	for p in T:
		#T[p] = partition(T[p])
		for i in range(len(T[p])-1):
			segment = [T[p][i], T[p][i+1]]
			L.append([segment,p,0])
	return L

def detect(T, L, D, P):
	outlier_count = 0
	c = 0
	for Li in L:
		distances = []
		CTR_count = 0
		for p in T:
			if p != Li[1]:
				matchLen = 0
				for i in range(len(T[p])-1):
					segment = [T[p][i], T[p][i+1]]
					#print(segment, p)
					dist = tp_distance(Li[0], segment)
					distances.append(dist)
					if dist < D:
						matchLen = matchLen + length(segment)
				if matchLen > length(Li[0]):
					CTR_count = CTR_count + 1
		# Calculate the density
		sd = standard_deviation(distances)
		density = (len([d for d in distances if d <= sd]) + 1)/len(L)
		# print(CTR_count, density, CTR_count/density, P*len(T))
		if CTR_count/density < P*len(T):
			Li[2] = 1
			outlier_count += 1
			# print("Outlying Segment Found: ", Li)
		c = c + 1
		# print(np.mean(distances))
		# print(c)
	return L, outlier_count

def mark(T, L, F):
	otraj = []
	for p in T:
		# If the ratio of length of outlying t-partitions to total length of trajectory is greater than F
		# Then mark this trajectory as outlying
		outliers = [Li[0] for Li in L if (Li[2] == 1 and Li[1] == p)]
		olen, tlen = 0, 0
		for seg in outliers:
			olen = olen + length(seg)
		for i in range(len(T[p])-1):
			segment = [T[p][i], T[p][i+1]]
			tlen = tlen + length(segment)
		if olen/tlen > F:
			otraj.append(p)
		# print(p)
	return otraj

# The outlier detection algorithm
def traod(T, D, P, F):
	# PARTITION PHASE
	# Partition each trajectory and store the t-partitions in array L
	print("Partition Phase Begins ...")
	L = partition(T)
	print("Partition Done !")
	print("Total Number of t-partitions: ", len(L))
	print("Outlying t-partition Detection Phase Begins ...")
	# DETECTION PHASE
	# For each t-partition in L count the number of trajectories that are close to it
	L, outlier_count = detect(T, L, D, P)
	print("Outlying t-partition Detection Done !")
	print("Number of Outlying t-partitions: ", outlier_count, " of ", len(L))
	print("Outlying Trajectory Detection Phase Begins ...")
	# MARKING PHASE
	otraj = mark(T, L, F)
	print("Outlying Trajectory Detection Phase Done !")
	print("Number of Outlying Trajectories: ", len(otraj), " of ", len(T))
	return otraj


###################################################################################################
# DATA EXTRACTION UNIT

import csv

# -------------------------------------------------------------------------------------------------

filename = 'data/csv/al_position2013-02-06.csv'

# This method converts hh:mm:ss to real valued time instant
def gettime(tr):
    return ((int(tr[0:2])*3600) + (int(tr[3:5])*60) + (int(tr[6:8])))/86400.0

# This method returns a list of the trajectories of the pedestrains in the data
def trajectory(filename, N):
    trajectory = {}
    csvfile = open(filename, newline='')
    data = csv.reader(csvfile, delimiter=' ', quotechar='|')
    xs, ys, ts = [], [], []
    t1, m1, x1, y1, p1 = -1, '', -1, -1, -1
    P = 0
    for row in data:
        r = ', '.join(row).split(";")
        t2, m2, x2, y2, p2 = gettime((r[0].split('T')[1])[:-4]), r[1], int(int(r[2])/67), int(int(r[3])/67), int(r[4])
        if (p1 == p2 and t1 != t2):
            if trajectory[P][len(trajectory[P])-1][0] != x2 or trajectory[P][len(trajectory[P])-1][1] != y2:
                trajectory[P].append([x2,y2])
        if p1 != p2:
            P = P + 1
            if not P <= N:
                break
            trajectory[P] = [[x2,y2]]
        t1, m1, p1, x1, y1 = t2, m2, p2, x2, y2
    return trajectory

def plot_trajectory(traj, p):
    x, y, t = [], [], []
    i = 1
    for point in traj:
        x.append(point[0])
        y.append(point[1])
        i = i + 1
    #     t.append(i)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x,y,t)
    # plt.show()
    plt.plot(x, y)
    # plt.show()
    fig.suptitle('TRAJECTORY ID: ' + str(p))
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig(str(p) + '.png')
# Synthetic data generation method
# def addbad_trajectories(T):
# 	i = len(T)+1
# 	return T


###################################################################################################
# EXECUTION SECTION

T = trajectory(filename, 10)
O = traod(T, 20, 0.01, 0.4)
print("Outliers: ", O)
###################################################################################################
# RESULTS PLOTTING SECTION
for p in T:
	plot_trajectory(T[p], p)
