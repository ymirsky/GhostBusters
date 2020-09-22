from GB_model import *

#Paths to images in npy format (dir contains subdirectories, one for each expert)
#       (provide the parent directory to the dataset generator's output)
data_path  = 'data/<path_here>'

# Load model
GB = GhostBusters(model_path='models', device_ID='0')

# predict
pred, filenames = GB.predict(path=data_path)

import numpy as np
print("%",100*np.sum(pred[:,1]>0.5)/len(pred),"of the samples were predicted as 'fake'.")