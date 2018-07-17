import cv2
import os
import numpy as np
import imageio


def watch_cut(avi_path, out_path, wait_length=20, clip_length=100):
    cap = cv2.VideoCapture(avi_path)

    while not cap.isOpened():
        cap = cv2.VideoCapture(avi_path)
        cv2.waitKey(1000)

    dir_path = os.path.join(out_path, os.path.basename(avi_path).split('.')[0])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        print("Directory already exists. Why???")
        return

    clip_idx = 0
    while True:
        clip_idx += 1
        path = os.path.join(dir_path, str(clip_idx))
        os.mkdir(path)
        frame_idx = 0
        recording = False

        while True:
            flag, frame = cap.read()
            key = cv2.waitKey(wait_length)
            if key == ord('a'):
                frame_1_path = os.path.join(path, str(frame_idx) + '.png')                 
                imageio.imsave(frame_1_path, frame[:, :, ::-1])
                recording = True
                frame_idx += 1
            elif frame_idx > clip_length:
                break
            elif recording == True:
                frame_1_path = os.path.join(path, str(frame_idx) + '.png')
                imageio.imsave(frame_1_path, frame[:, :, ::-1])
                frame_idx += 1
            elif recording == False and key == ord('b'):
                cap.set(int(cv2.CAP_PROP_POS_FRAMES),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 20))

            if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
                return

            ### DEBUG
            print(clip_idx + 1, cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_POS_FRAMES))

