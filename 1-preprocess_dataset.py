from GB_extractor import *

# load extractor and init object detector model
E = Extractor(device_ID="0") #using GPU 0

# Extract all signs from frames in the video
# will save all signs found in img_path into 4 +1 subdirectories: 
#    <save_dir>/context
#    <save_dir>/surface
#    <save_dir>/light
#    <save_dir>/optical
#    <save_dir_nosign>/context
path_to_video = "/home/user/video.mp4" #example
save_dir = "data/real" # or "data/fake"
save_dir_nosign = "data/real_nosign" #note, the source of nosign examples should be from real signs 
E.extract_from_video(path_to_video, save_dir, True, save_dir_nosign)


# If you don't want to generate samples of 'nosign', then use this:
E.extract_from_video(path_to_video, save_dir)
