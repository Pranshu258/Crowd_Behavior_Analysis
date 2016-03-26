# Module: outlierDetector
# Defines methods to detect outliers in a given set of trajectories
# References: 
# Trajectory Outlier Detection: A Partition-and-Detect Framework, Lee, Han, Li
# Trajectory Clustering: A Partition-and-Group Framework, Jae-Gil Lee, Jiawei Han, Kyu-Young Whang
# -------------------------------------------------------------------------------------------------

import numpy as np, math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def tp_distance(L1, L2):
	# print("L1 = ", L1, "L2 = ", L2)
	s1x, s1y, e1x, e1y = L1[0][0], L1[0][1], L1[1][0], L1[1][1]
	s2x, s2y, e2x, e2y = L2[0][0], L2[0][1], L2[1][0], L2[1][1]
	D = np.linalg.norm([s2y-s1y, s2x-s1x]) + np.linalg.norm([e2y-e1y, e2x-e1x])
	return D

# This method returns the length of a t-partition
def length(segment):
	sx, sy, ex, ey = segment[0][0], segment[0][1], segment[1][0], segment[1][1]
	return np.linalg.norm([ex-sx, ey-sy])

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
	for Li in L:
		distances = []
		CTR_count = 0
		for p in T:
			if p != Li[1]:
				matchLen = 0
				for i in range(len(T[p])-1):
					segment = [T[p][i], T[p][i+1]]
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
	return otraj

# The outlier detection algorithm
def traod(T, D, P, F):
	# PARTITION PHASE
	# Partition each trajectory and store the t-partitions in array L
	print("Partition Phase Begins ...")
	L = partition(T)
	print("Partition Done !")
	print("Total Number of Segments: ", len(L))
	print("Outlying Segment Detection Phase Begins ...")
	# DETECTION PHASE
	# For each t-partition in L count the number of trajectories that are close to it
	L, outlier_count = detect(T, L, D, P)
	print("Outlying Segment Detection Done !")
	print("Number of Outlying Segments: ", outlier_count, " of ", len(L))
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
        if (p1 == p2 and (x1 != x2 or y1 != y2) and t1 != t2):
            trajectory[P].append([x2,y2])
        if p1 != p2:
            P = P + 1
            if not P <= N:
            	break
            trajectory[P] = [[x2,y2]]
        t1, m1, x1, y1, p1 = t2, m2, x2, y2, p2
    return trajectory

def plot_trajectory(traj):
    x, y, t = [], [], []
    i = 1
    for point in traj:
        x.append(point[0])
        y.append(point[1])
        i = i + 1
        t.append(i)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,t)
    plt.show()

# Synthetic data generation method
# def addbad_trajectories(T):
# 	i = len(T)+1
# 	return T


###################################################################################################
# EXECUTION SECTION

T = trajectory(filename, 50)
# O = traod(T, 50, 0.01, 0.4)

###################################################################################################
# RESULTS PLOTTING SECTION
# for p in O:
# 	plot_trajectory(T[p])
