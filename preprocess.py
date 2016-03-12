#!/usr/bin/python3
# =============================================================================
# Crowd Behavior Ananlysis
# Author: Pranshu Gupta, Lavisha Aggarwal
# =============================================================================

import scipy, csv, pickle, os

# -----------------------------------------------------------------------------
rootDir = 'data/csv/'
# -----------------------------------------------------------------------------

def getData(filename):
	csvfile = open(filename, newline='')
	data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	dump = {}
	for row in data:
		r = ', '.join(row).split(";")
		t, m, x, y, p = (r[0].split('T')[1])[:-4], r[1], int(int(r[2])/(3*6700)), int(int(r[3])/(3*6700)), int(r[4])
		if t not in dump:
			dump[t] = {(x,y):set([p])}
			# print(len(dump))
		else:
			if (x,y) not in dump[t]:
				dump[t][(x,y)] = set([])
			dump[t][(x,y)].add(p)
	return dump

def dumpData(rootDir):
	files = os.listdir(rootDir)
	i = 0
	for filename in files:
		fname = rootDir + filename
		data = getData(fname)
		dumpfile = open('data/frames/day' + str(i) + '.pickle', 'wb')
		pickle.dump(data, dumpfile, pickle.HIGHEST_PROTOCOL)
		print('day ' + str(i) + ' done')
		i = i + 1

# -----------------------------------------------------------------------------

# Preprocessing of the data
dumpData(rootDir)
