import math
from sklearn import preprocessing
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.misc import imresize


#Calculates diffusion distance of two histograms(given as parameters)
def diffusionDistance(h1, h2):
	shape = list(h1.shape)
	
	#d_0 is the difference between two histograms
	d_prev = np.subtract(h1, h2)
	
	#K at t=0 is L1 norm of d_0
	K = np.sum(np.absolute(d_prev))
	
	#Till the d matrix downsamples to size 1
	while(min(shape) > 1):
	
		#Applying gaussian filter over d_prev before downsampling
		d = gaussian_filter(d_prev, sigma = 0.05)
	
		#downsampling upto 50%
		d = imresize(d, 0.5, interp='nearest')
		
		#Summing the L1 norm of d to K
		K = K + np.sum(np.absolute(d))
		shape = list(d.shape)
		d_prev = d
	#Outputting the diffusion distance
	print(K)
	
	
#Example:

import matplotlib.pyplot as plt
import numpy as np

import plotly.plotly as py


h1 = [[1, 0], [1, 0]]
h2 = [[0, 1], [0, 1]]
h3 = [[0, 2], [0, 0]]
#converting matrices to numpy arrays
h1 = np.asarray(h1)
h2 = np.asarray(h2)
h3 = np.asarray(h3)

#We see that h1 and h2 are more similar than h1 and h3.

plt.hist(h1.flatten(),2,[0,2], color = 'b')
plt.xlim([0,2])
plt.title('histogram1')
plt.show()

plt.hist(h2.flatten(),2,[0,2], color = 'b')
plt.xlim([0,2])
plt.title('histogram2')
plt.show()

plt.hist(h3.flatten(),2,[0,2], color = 'b')
plt.xlim([0,2])
plt.title('histogram3')
plt.show()

#EMD distance of h1 and h2 are same.

#Diffusion distance of :
#h1 and h2 -
print("Diffusion distance of h1 and h2 is ")
diffusionDistance(h1, h2)
#h1 and h3 -
print("Diffusion distance of h1 and h3 is ")
diffusionDistance(h1, h3)


