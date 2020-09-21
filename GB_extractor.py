import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, sys, traceback
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
import glob as glob

from object_detection.utils import label_map_util #conda install -c conda-forge tf_object_detections
from object_detection.utils import visualization_utils as vis_util

import cv2 as cv
import time
#from matplotlib import pyplot as plt


class Extractor:
    def __init__(self,device_ID=None):
        # Device ID is the GPU index to use. Set to "" for CPU and leave as None for default
        if device_ID is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=device_ID
            
        # The model used to identify road signs
        self.MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous'
        # Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
        self.MODEL_PATH = os.path.join('models', self.MODEL_NAME)
        self.PATH_TO_CKPT = os.path.join(self.MODEL_PATH,'inference_graph/frozen_inference_graph.pb')

        # load the object detector
        print("Loading",self.MODEL_NAME,"...")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.categories = ["prohibitory","mandatory","danger"]
        self.category_index = {1:{'name':"prohibitory"},2:{'name':"mandatory"},3:{'name':"danger"}}
        self.label_map={"prohibitory":1,"mandatory":2,"danger":3}
        
        self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
        
    
    
    def extract_from_image(self, img_path, save_dir, make_nosign_examples=False, nosign_save_dir=None):
        # will save all signs found in img_path into 4 subdirectories: <save_dir>/context, <save_dir>/surface, ...
        print('WARNING: Cannot extract features for Depth Expert (optical flow) without a video.')
        if make_nosign_examples and (nosign_save_dir is None):
            raise Exception('You must provide a save directory for the nosign examples.')
        if not os.path.exists(vid_path):
            raise Exception('Could not find video: '+vid_path)
        
        # make save directories (to save time, we assume that if context is there then all other are too)
        save_dir_c = os.path.join(save_dir,'context')
        if not os.path.exists(save_dir_c):
            os.makedirs(save_dir_c)

            save_dir_s = os.path.join(save_dir,'surface')
            if not os.path.exists(save_dir_s):
                os.makedirs(save_dir_s)
            save_dir_l = os.path.join(save_dir,'light')
            if not os.path.exists(save_dir_l):
                os.makedirs(save_dir_l)
            save_dir_o = os.path.join(save_dir,'optical')
            if not os.path.exists(save_dir_o):
                os.makedirs(save_dir_o)
            if make_nosign_examples:
                save_dir_c_ns = os.path.join(nosign_save_dir,'context')
                if not os.path.exists(save_dir_c_ns):
                    os.makedirs(save_dir_c_ns)
                

        
        # Search for roadsigns...
        image_np = np.array(cv.imread(img_path,0))
        image_np_last = np.copy(image_np)
        if image_np is None:
            raise Exception('Error: could not read',img_path)
                
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        try:
            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            if frame_idx == 1:
                print("Model Loaded.")
            for box in boxes[scores>0.7]:
                try:
                    img2=self.crop_scaled(image_np,box,2.5,128)
                    img2_last=self.crop_scaled(image_np_last,box,2.5,128)
                    img1=self.crop_scaled(image_np,box,1.0,128)
                    if (img2 is not None) and (img1 is not None) and (img2_last is not None):                             
                        ### extract: context, surface, light, and depth features ###
                        img_c, img_s, img_l, img_o = self.extract_features(img1,img2,img2_last)
                        count+=1
                        file_prefix =  str(np.abs(hash(vid_path)))[:6] + '_' + str(count) + '_'
                        np.save(os.path.join(save_dir_c, file_prefix + 'c.npy'),img_c)
                        np.save(os.path.join(save_dir_s, file_prefix + 's.npy'),img_s)
                        np.save(os.path.join(save_dir_l, file_prefix + 'l.npy'),img_l)
                        np.save(os.path.join(save_dir_o, file_prefix + 'o.npy'),img_o)
                        if make_nosign_examples:
                            imgs_ns = self.extract_nosign(box,image_np)
                            for img_ns in imgs_ns:
                                np.save(os.path.join(save_dir_c_ns, file_prefix + 'ns_c.npy'),img_ns)
                except:
                    print('-'*60)
                    print('Failed: to process or save detected sign:')
                    traceback.print_exc(file=sys.stdout)
                    print('-'*60)
        except:
            print('Failed: to run model on image')
        
            
    def extract_from_video(self, vid_path, save_dir, make_nosign_examples=False, nosign_save_dir=None):
        # will save all signs found in vid_path into 4 subdirectories: <save_dir>/context, <save_dir>/surface, ...
        if make_nosign_examples and (nosign_save_dir is None):
            raise Exception('You must provide a save directory for the nosign examples.')
        if not os.path.exists(vid_path):
            raise Exception('Could not find video: '+vid_path)
        
        # make save directories
        save_dir_c = os.path.join(save_dir,'context')
        if not os.path.exists(save_dir_c):
            os.makedirs(save_dir_c)
        save_dir_s = os.path.join(save_dir,'surface')
        if not os.path.exists(save_dir_s):
            os.makedirs(save_dir_s)
        save_dir_l = os.path.join(save_dir,'light')
        if not os.path.exists(save_dir_l):
            os.makedirs(save_dir_l)
        save_dir_o = os.path.join(save_dir,'optical')
        if not os.path.exists(save_dir_o):
            os.makedirs(save_dir_o)
        if make_nosign_examples:
            save_dir_c_ns = os.path.join(nosign_save_dir,'context')
            if not os.path.exists(save_dir_c_ns):
                os.makedirs(save_dir_c_ns)
        
        # Search for roadsigns...
        print('Working on video',vid_path)
        vidcap = cv.VideoCapture(vid_path)
        num_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        success = True
        frame_idx = 0
        image_np_last=None
        count=0
        while success:
            frame_idx +=1
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            success,image = vidcap.read()
            image_np = np.array(image) 
            if image_np is None:
                continue
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            try:
                (boxes, scores, classes, num_detections) = self.sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                if frame_idx == 1:
                    print("Model Loaded.")
                for box in boxes[scores>0.7]:
                    try:
                        img2=self.crop_scaled(image_np,box,2.5,128)
                        img2_last=self.crop_scaled(image_np_last,box,2.5,128)
                        img1=self.crop_scaled(image_np,box,1.0,128)
                        if (img2 is not None) and (img1 is not None) and (img2_last is not None):                             
                            ### extract: context, surface, light, and depth features ###
                            img_c, img_s, img_l, img_o = self.extract_features(img1,img2,img2_last)
                            count+=1
                            file_prefix =  str(np.abs(hash(vid_path)))[:6] + '_' + str(count) + '_'
                            np.save(os.path.join(save_dir_c, file_prefix + 'c.npy'),img_c)
                            np.save(os.path.join(save_dir_s, file_prefix + 's.npy'),img_s)
                            np.save(os.path.join(save_dir_l, file_prefix + 'l.npy'),img_l)
                            np.save(os.path.join(save_dir_o, file_prefix + 'o.npy'),img_o)
                            if make_nosign_examples:
                                imgs_ns = self.extract_nosign(box,image_np)
                                for img_ns in imgs_ns:
                                    np.save(os.path.join(save_dir_c_ns, file_prefix + 'ns_c.npy'),img_ns)
                    except:
                        print('-'*60)
                        print('Failed: to process or save detected sign:')
                        traceback.print_exc(file=sys.stdout)
                        print('-'*60)
            except:
                print('Failed: to run model on image')
            image_np_last = image_np.copy()
            
            if (frame_idx-1)%100==0:
                print("Progress:",np.round(100*frame_idx/num_frames,1),"%"," #Signs found:",count)
        print("Dataset Extraction complete.")
        
    def extract_features(self,img1,img2,img2_last):
        # Context
        img_c = np.copy(img2)
        w=45; d=img_c.shape[0]
        img_c[(w):(d-w),(w):(d-w),:] = 0 # erase center
        img_c = img_c.astype('float')
        img_c /= 255.0
        
        # Surface
        img_s = np.copy(img1)
        m=20; d=img_s.shape[0]
        img_s = img_s[m:(d-m),m:(d-m),:] #crop
        img_s = img_s.astype('float') 
        img_s /= 255.0
        
        # Light
        img_l = np.copy(img1)
        m=20; d=img_l.shape[0]
        img_l = img_l[m:(d-m),m:(d-m),:] #crop
        img_l = np.max(img_l,axis=2) # light
        img_l = np.expand_dims(img_l,-1) # 1 channel
        img_l = img_l.astype('float') 
        img_l /= 255.0
        
        # Depth (optical flow)
        flow = cv.calcOpticalFlowFarneback(cv.cvtColor(img2_last,cv.COLOR_BGR2GRAY),cv.cvtColor(img2,cv.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(img2_last)
        hsv[:,:,1] = 255
        hsv[:,:,0] = ang*180/np.pi/2
        hsv[:,:,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        img_o = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        img_o = img_o.astype('float') 
        img_o /= 255.0

        return img_c, img_s, img_l, img_o
    
    def extract_nosign(self,box,image_np):
        # take 3 random plausble locations that are NOT box (the deteted sign)
        boxf1 = np.array([box[0],1-box[1],box[2],1-box[1]+(box[3]-box[1])])
        if box[1] < 0.5:  #left side
            boxf2 = np.array([boxf1[0],boxf1[1]+(box[3]-box[1]),boxf1[2],boxf1[3]+(box[3]-box[1])])
            boxf3 = np.array([boxf1[0],boxf1[1]+2*(box[3]-box[1]),boxf1[2],boxf1[3]+2*(box[3]-box[1])])
        else: #right
            boxf2 = np.array([boxf1[0],boxf1[1]-(box[3]-box[1]),boxf1[2],boxf1[3]-(box[3]-box[1])])
            boxf3 = np.array([boxf1[0],boxf1[1]-2*(box[3]-box[1]),boxf1[2],boxf1[3]-2*(box[3]-box[1])])
                
        imgs_ns = []
        img1_ns = self.crop_scaled(image_np,boxf1,2.5,128)
        if img1_ns is not None:
            w=45; d=img1_ns.shape[0]
            img1_ns[(w):(d-w),(w):(d-w),:] = 0 # erase center
            img1_ns = img1_ns.astype('float') 
            img1_ns /= 255.0
            imgs_ns.append(img1_ns)

        img2_ns = self.crop_scaled(image_np,boxf2,2.5,128)
        if img2_ns is not None:
            w=45; d=img2_ns.shape[0]
            img2_ns[(w):(d-w),(w):(d-w),:] = 0 # erase center
            img2_ns = img2_ns.astype('float') 
            img2_ns /= 255.0
            imgs_ns.append(img2_ns)

        img3_ns = self.crop_scaled(image_np,boxf3,2.5,128)
        if img3_ns is not None:
            w=45; d=img3_ns.shape[0]
            img3_ns[(w):(d-w),(w):(d-w),:] = 0 # erase center
            img3_ns = img3_ns.astype('float') 
            img3_ns /= 255.0
            imgs_ns.append(img3_ns)
            
        return imgs_ns
    
    def get_label_id(self,label_name):
        for category in categories:
            if category['name'] == label_name:
                return category['id']
        
    def crop_scaled(self,image, box, bscale = 1.5, dim=128):
        image = np.copy(image)
        box = np.copy(box)
        dh=box[2]-box[0]
        dw=box[3]-box[1]
        box[0]-=dh*(bscale-1)
        box[2]+=dh*(bscale-1)
        box[1]-=dw*(bscale-1)
        box[3]+=dw*(bscale-1)
        if (np.abs(box[0]-0.5)>0.5) or (np.abs(box[1]-0.5)>0.5) or (np.abs(box[2]-0.5)>0.5) or (np.abs(box[3]-0.5)>0.5): #out of bounds of image
            h = 256/image.shape[0] 
            w = 256/image.shape[1]
            image1 = np.zeros((image.shape[0]+500,image.shape[1]+500,3),dtype='uint8')
            image1[256:(image.shape[0]+256),256:(image.shape[1]+256),:] = image
            image = image1 
            box[0] = (box[0]+h)/(1+2*h)       
            box[1] = (box[1]+w)/(1+2*w)       
            box[2] = (box[2]+h)/(1+2*h)       
            box[3] = (box[3]+w)/(1+2*w)       

        dims = np.array([box[0]*image.shape[0],box[1]*image.shape[1],box[2]*image.shape[0],box[3]*image.shape[1]],dtype=int)
        img = image[dims[0]:dims[2],dims[1]:dims[3],:]
        try:
            img = cv.resize(img, dsize=(dim, dim), interpolation=cv.INTER_CUBIC)
            return img
        except:
            return None
    
        
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
