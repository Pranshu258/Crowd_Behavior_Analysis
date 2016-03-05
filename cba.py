#!/usr/bin/python3
# =============================================================================
# Crowd Behavior Ananlysis
# Author: Pranshu Gupta, Lavisha Aggarwal
# =============================================================================

import numpy as np
import scipy
import cv2

# -----------------------------------------------------------------------------
# Hamming Distance
def hamming(u, v):
	"""hamming : computes the hamming distance between two bit-vectors"""
	return scipy.spatial.distance.hamming(u, v)

# Social Affinity Map
def sam():
	"""sam : computes the social affinity map for a given tracklet"""
	
# -----------------------------------------------------------------------------