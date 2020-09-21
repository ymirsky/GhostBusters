import argparse

parser = argparse.ArgumentParser(description='A CLI tool for GhostBusters, for preparing datasets, training, and execution (prediction)\nFor more information, please see our paper:\n Ben Nassi, Yisroel Mirsky, ... Phantom of the ADAS: Securing Advanced Driver-AssistanceSystems from Split-Second Phantom Attacks, CCS 2021. Tool developed by Yisroel Mirsky.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Actions
group1 = parser.add_argument_group('Procedures', 'Use one or more of these flags')
group1.add_argument('-pr', '--proc_real', action="store",  metavar='<vid_path>', help='Extract a real-sign dataset from the given video. Will automatically extract nosign examples. \nE.g., $ python GhostBusters.py -pr /home/user/real_vid.mp4')
group1.add_argument('-pf', '--proc_fake', action="store", metavar='<vid_path>',help='Extract a fake-sign dataset from the given video')
group1.add_argument('-t', '--train', action="store_true", help='Extract a dataset from a video: The given path')
group1.add_argument('-e', '--predict', action="store", metavar='<data_dir>',help='Predict on the given path. The path must be to a directory containing four subdirectories of the preprocessed signs (context, surface, light, optical). Outputs two columns: col0 is probability of being real, col1 is probability of being fake.')

# Parameters
group2 = parser.add_argument_group('Parameters', 'Configurable parameters (optional)')
group2.add_argument('-dd', '--data_dir', action="store", default='data/', metavar='<data_dir>', help='Set the save/load data directory.')
group2.add_argument('-md', '--model_dir', action="store", default='models/', metavar='<model_dir>', help='Set the save/load model directory.')
group2.add_argument('--exp_epochs', action="store", type=int, default=25, metavar='E', help='Training eopochs for each expert.')
group2.add_argument('--com_epochs', action="store", type=int, default=10, metavar='E', help='Training eopochs for the combiner.')
group2.add_argument('--device_id', action="store", default="0", metavar='ID', help='The ID of the GPU to use: 0, 1, ... or "" for CPU.')
group2.add_argument('--pred_path', action="store",  default="predictions/pred.csv", metavar='<pred_dir>', help='Set the save path for the predictions, saved as a csv.')

args = parser.parse_args()

if (args.proc_real is None) and (args.proc_fake is None) and (not args.train) and (args.predict is None):
    parser.print_help()
    print('You must use at least one of the Procedure arguments.')
else:
    # Run GhostBusters:

    # Preprocess Data
    if (args.proc_real is not None) or (args.proc_fake is not None):
        import os
        from GB_extractor import *
        # load extractor and init object detector model
        E = Extractor(device_ID=args.device_id)
        if args.proc_real is not None:
            vid_path = args.proc_real
            save_dir =  os.path.join(args.data_dir,"real")
            save_dir_nosign = os.path.join(args.data_dir,"real_nosign")
            E.extract_from_video(vid_path, save_dir, True, save_dir_nosign)
        if args.proc_fake is not None:
            vid_path = args.proc_fake
            save_dir =  os.path.join(args.data_dir,"fake")
            E.extract_from_video(vid_path, save_dir)

    # Train model
    if args.train:
        from GB_model import *
        # Init model
        GB = GhostBusters(save_path=args.model_dir,device_ID=args.device_id)
        # Train model (first experts then combiner)
        real_path = os.path.join(args.data_dir,"real")
        fake_path = os.path.join(args.data_dir,"fake")
        nosign_path = os.path.join(args.data_dir,"real_nosign")
        GB.train(real_path,fake_path,nosign_path,epochs=[args.exp_epochs,args.com_epochs])

    # Predict on new signs
    if args.predict is not None:
        from GB_model import *
        # Load model
        GB = GhostBusters(model_path=args.model_dir, device_ID=args.device_id)
        # predict
        data_path = args.predict
        pred = GB.predict(path=data_path)
        import numpy as np
        print("%",100*np.sum(pred[:,1]>0.5)/len(pred),"of the samples were predicted as 'fake'.")
        if (args.pred_path == "predictions/pred.csv") and (not os.path.exists('predictions/')):
            os.makedirs('predictions/')
        np.savetxt(args.pred_path,pred,delimiter=',')

