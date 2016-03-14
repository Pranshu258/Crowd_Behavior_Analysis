#!/usr/bin/python3
# =============================================================================
# Crowd Behavior Analysis
# Author: Pranshu Gupta, Lavisha Aggarwal
# =============================================================================

import scipy, csv, pickle, os, numpy as np

# -----------------------------------------------------------------------------
rootDir = 'data/csv/'
# -----------------------------------------------------------------------------

# This method returns the presence count at each second
def getPresence(filename):
	csvfile = open(filename, newline='')
	data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	dump = {}
	for row in data:
		r = ', '.join(row).split(";")
		t, m, x, y, p = (r[0].split('T')[1])[:-4], r[1], int(int(r[2])/(6700)), int(int(r[3])/(3*6700)), int(r[4])
		if t not in dump:
			dump[t] = {(x,y):set([p])}
			# print(len(dump))
		else:
			if (x,y) not in dump[t]:
				dump[t][(x,y)] = set([])
			dump[t][(x,y)].add(p)
	return dump

# This method returns the flow situations for each second
def getFlow(filename):
	csvfile = open(filename, newline='')
	data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	dump = {}
	t, m, x, y, p = '', '', -1, -1, -1
	for row in data:
		r = ', '.join(row).split(";")
		x1, y1, p1 = x, y, p
		t, m, x, y, p = (r[0].split('T')[1])[:-4], r[1], int(int(r[2])/(6700)), int(int(r[3])/(3*6700)), int(r[4])
		if p1 == p and (x1 != x or y1 != y):
			# The person has moved to some other place
			if t not in dump:
				dump[t] = {(x1,y1):np.array([0,0,0,0,0,0,0,0])}
			else:
				if (x1,y1) not in dump[t]:
					dump[t][(x1,y1)] = np.array([0,0,0,0,0,0,0,0])
			# Now add the flow value to the histogram
			disp = (x-x1, y-y1)
			if disp == (0,-1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([1,0,0,0,0,0,0,0])
			elif disp == (1,-1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,1,0,0,0,0,0,0])
			elif disp == (1,0):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,1,0,0,0,0,0])
			elif disp == (1,1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,0,1,0,0,0,0])
			elif disp == (0,1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,0,0,1,0,0,0])
			elif disp == (-1,1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,0,0,0,1,0,0])
			elif disp == (-1,0):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,0,0,0,0,1,0])
			elif disp == (-1,-1):
				dump[t][(x1,y1)] = dump[t][(x1,y1)] + np.array([0,0,0,0,0,0,0,1])
			else:
				print("Bigger Movement")
	return dump

# -----------------------------------------------------------------------------
