from GB_model import *

#Paths to images of real, fake (phantom), and 'no signs' in npy format (each dir contains subdirectories, one for each expert)
#       (provide the parent directory to the dataset generator's output for real, fake, and nosign)
real_path  = 'data/real'
fake_path  = 'data/fake'
nosign_path = 'data/real_nosign'

# Init model
GB = GhostBusters(save_path='models',device_ID="0")

# Train model (first experts then combiner)
GB.train(real_path,fake_path,nosign_path)


