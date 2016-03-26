# Module: outlierDetector
# Defines methods to detect outliers in a given set of trajectories
# References: 
# Trajectory Outlier Detection: A Partition-and-Detect Framework, Lee, Han, Li
# Trajectory Clustering: A Partition-and-Group Framework, Jae-Gil Lee, Jiawei Han, Kyu-Young Whang
# -------------------------------------------------------------------------------------------------

import numpy as np, math

# -------------------------------------------------------------------------------------------------
# DEFINITIONS

# CTR_count: count of the trajectories close to the t-partition Li
# D: parameter given by user, the threshold for closeness of two t-partitions
# F: parameter given by user, the threshold for marking a trajectory as outlier
# p: parameter given by user, the factor used in marking a t-partition as outlier 
# density: The count of t-partitions that are within sd distance to the given t-partition

# -------------------------------------------------------------------------------------------------
# METHODS

# Distance between two t-partitions
# This is not the distance measure used in the paper by Lee et. al.
def tp_distance(L1, L2):
	s1x, s1y, e1x, e1y = L1[0][0], L1[0][1], L1[1][0], L1[1][1]
	s2x, s2y, e2x, e2y = L2[0][0], L2[0][1], L2[1][0], L2[1][1]
	D = np.linalg.norm([s2y-s1y, s2x-s1x]) + np.linalg.norm([e2y-e1y, e2x-e1x])
	return D

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
		# if the point lies within the line segment

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
def partition(t):
	cp = [t[0]]
	si, l = 1, 1
	while si + l <= len(t):
		ci = si + l
		cost_par = mdl_par(t, si,ci)
		cost_nopar = mdl_nopar(t, si,ci)
		# If partitioning cost is greater than the no-partitioning cost, keep the original point
		if cost_par > cost_nopar:
			cp.append(t[ci-1])
			si, l = ci-1, 1
		else:
			l = l + 1
	cp.append(t[-1])

# The outlier detection algorithm
def traod(T, D, p, F):
	# PARTITION PHASE
	# Partition each trajectory and store the t-partitions in array L
	L = []
	for p in T:
		T[p] = partition(T[p])
		for i in range(len(T[p])-1):
			segment = [T[p][i], T[p][i+1]]
			L.append((segment,p,0))
	# DETECTION PHASE
	# For each t-partition in L count the number of trajectories that are close to it
	for Li in L:
		distances = []
		for p in T:
			if p != Li[1]:
				matchLen = 0
				for i in range(len(T[p])-1):
					segment = [T[p][i], T[p][i+1]]
					dist = tp_distance(Li, segment)
					distances.append(dist)
					if dist < D:
						matchLen = matchLen + length(segment)
				if matchLen > length(Li[0]):
					CTR_count = CTR_count + 1
		# Calculate the density
		sd = standard_deviation(distances)
		density = len([d for d in distances if d <= sd])
		if CTR_count/density < p*len(T):
			Li[2] = 1
	# MARKING PHASE
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
