import cv2
import numpy as np
from matplotlib import pyplot as plt

from os import listdir
path = "RD-cvpr06"
files = [ path+ "/" + f for  f in listdir(path)]
#print(files)

for filename in files : 
#filename = 'RD-cvpr06/r0101.jpg'
	img = cv2.imread(filename)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	x = np.array(dst)


	dst = cv2.dilate(dst,None)
	indices =  np.argpartition(x.flatten(), -100)[-100:]
	indices = np.vstack(np.unravel_index(indices, x.shape)).T
	s = x.shape
	x2 = np.zeros(x.shape, dtype=bool)
	for l in indices :
		x2[l[0]][l[1]] = True

	print("No of interest points " +filename+" considered : " + str(np.sum(x2)))
	img[x2]=[255,0,0]

	plt.imshow(img,cmap = 'gray')
	plt.title('Harris corner Detection'+filename), plt.xticks([]), plt.yticks([])
	plt.show()
