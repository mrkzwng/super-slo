import cv2
import zipfile
import re
import os
from os.path import split
from pathlib import Path


filedir = '../data/DeepVideoDeblurring_Dataset.zip'
destdir = '../data/triplets/'
get_jpg_name = lambda path: split(split(split(path)[0])[0])[1] \
                    + '_' + os.path.basename(path)
get_parent_name = lambda path: split(split(split(path)[0])[0])[1]
# generate .jpg files
with zipfile.ZipFile(filedir, 'r') as z:
    names = z.namelist()
    names = [name for name in names 
             if re.search('^[^_MACOSX].*(input|GT).*\.jpg', name)]
    jpg_names = [get_jpg_name(name) for name in names]
    for jpg_name, zip_path in zip(jpg_names, names):
        with open(os.path.join(destdir, jpg_name), 'wb') as f:
            f.write(z.read(zip_path))

parent_names = set([get_parent_name(name) for name in names])
for par_name in sorted(parent_names):
    vid_frames = [path + '\n' for path in new_paths
                  if par_name in path]
    for frame_idx in range(1, 10):
        fr_path = os.path.join(listdir+'frame'+str(frame_idx)+'.txt')
        if not os.path.isfile(fr_path):
            with open(fr_path, 'w') as f:
                f.writelines(
                    vid_frames[frame_idx-1:len(vid_frames)-(9-frame_idx)])
        else:
            with open(fr_path, 'a') as f:
                f.writelines(
                    vid_frames[frame_idx-1:len(vid_frames)-(9-frame_idx)])