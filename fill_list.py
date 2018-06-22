import cv2
import zipfile
import re
import os
from os.path import split
from pathlib import Path


zipdir = '../data/DeepVideoDeblurring_Dataset.zip'
destdir = '../data/triplets/'
listdir = './data_list/'
get_jpg_name = lambda path: split(split(split(path)[0])[0])[1] \
                    + '_' + os.path.basename(path)
get_parent_name = lambda path: split(split(split(path)[0])[0])[1]
# generate .jpg files
with zipfile.ZipFile(zipdir, 'r') as z:
    names = z.namelist()
    names = [name for name in names 
             if re.search('^[^_MACOSX].*(input|GT).*\.jpg', name)]
    jpg_names = [get_jpg_name(name) for name in names]
    new_paths = [os.path.join(destdir, jpg_name)
                 for jpg_name in jpg_names]
    for jpg_name, zip_path, new_path in zip(jpg_names, names, new_paths):
        with open(new_path, 'wb') as f:
            print("Writing "+jpg_name+" to "+new_path)
            f.write(z.read(zip_path))
# write .jpg paths to another file for loading
print("Writing .jpg paths to "+listdir)
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