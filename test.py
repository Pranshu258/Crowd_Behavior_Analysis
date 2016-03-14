def dumpData(rootDir):
	files = os.listdir(rootDir)
	i = 0
	for filename in files:
		fname = rootDir + filename
		data = getPresence(fname)
		dumpfile = open('data/frames/day' + str(i) + '.pickle', 'wb')
		pickle.dump(data, dumpfile, pickle.HIGHEST_PROTOCOL)
		print('day ' + str(i) + ' done')
		i = i + 1