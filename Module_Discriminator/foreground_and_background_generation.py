import numpy as np
from PIL import Image
import os
import cv2 as cv
from collections import defaultdict
from pprint import pprint
import re

data_dir = "/Users/saumya/Downloads/Mask_RCNN_results/ue_airsim"

data_dict=defaultdict(dict)
list_files = os.listdir("/Users/saumya/Desktop/texture_for_discriminator")
list_files= [x[:-4] for x in list_files]
print(list_files)

list_seg_scene = os.listdir(data_dir)
for img in list_files:
    data_dict[img]["seg"]=[]
    for file in list_seg_scene:
        if file.startswith(img) and file.endswith("_seg.jpg"):
            data_dict[img]["seg"].append(os.path.join(data_dir, file))
        if file.startswith(img) and file.endswith("_scene.jpg"):
            data_dict[img]["scene"] = os.path.join(data_dir, file)


# print(data_dict)

text_file = open("/Users/saumya/Desktop/Camouflage_Project/score.txt", "r")
lines = text_file.readlines()
for img in data_dict:
    for line in lines:
        line.strip()
        if line.startswith(img):
            temp = line.rsplit(",",1)
            temp = temp[1][1:-2]
            # print(temp)
            if temp!="":
                data_dict[img]["score"] = float(temp)
            else:
                data_dict[img]["score"] = 0.0

pprint(data_dict)

saving_dir = "/Users/saumya/Desktop/Camouflage_Project/training_data/"
for img in data_dict:

    scene_img_path = data_dict[img]["scene"]

    for background_path in data_dict[img]["seg"]:
        # print(background_path)
        background_seg = cv.imread(background_path)
        cv.imwrite(saving_dir + "seg_mask.jpg", background_seg)
        scene_img = cv.imread(scene_img_path)
        lower = np.array([135, 75, 235])
        upper = np.array([155, 95, 255])

        shapeMask = cv.inRange(background_seg, lower, upper)
        inds = np.where(shapeMask>0)
        background_seg[shapeMask>0] = (255,255,255)
        background_seg[shapeMask <= 0] = (0, 0, 0)
        cv.resize(background_seg, (64,64))

        scene_img[inds] = (0,0,0)
        cv.resize(scene_img, (64, 64))

        cv.imwrite(saving_dir+"foreground_mask.jpg", background_seg)
        cv.imwrite(saving_dir + "back_ground_mask.jpg", scene_img)
        # print(background_seg.shape)
