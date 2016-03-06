#!/usr/bin/python3
# =============================================================================
# Crowd Behavior Ananlysis
# Author: Pranshu Gupta, Lavisha Aggarwal
# =============================================================================

import scipy
import csv

# -----------------------------------------------------------------------------

# Hamming Distance
def hamming(u, v):
	"""hamming : computes the hamming distance between two bit-vectors"""
	return scipy.spatial.distance.hamming(u, v)

# Social Affinity Map
def sam():
	"""sam : computes the social affinity map for a given tracklet"""

# CSV Reader
def getData(filename):
	DATA = []
	csvfile = open(filename, newline='')
	data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in data:
		print(', '.join(row).split(";"))

# -----------------------------------------------------------------------------