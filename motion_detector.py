import numpy as np 
import cv2
import os
import psutil

''' TODO:
- check lossless codecs for _init_writer
- fix divide by 2 hacks for frame count and fps
'''

class MotionDetector:
    def __init__(self, threshold, out_path, frame_seq=3):
        """
        args:
        =threshold: threshold for throwing frames away
                    in mean pixel intensity
        =out_path: output path
        =frame_seq: minimum of number of consecutive 
                    frames to keep
        """
        self.threshold = threshold
        self.frame_seq = frame_seq
        self.out_path = out_path


    def _difference(self, old_frame, new_frame):
        """
        args:
        =old_frame: previous frame
        =new_frame: current frame

        returns:
        mean absolute pixel-wise intensity difference
        """
        return(np.mean(np.absolute(old_frame - new_frame)))


    def _init_writer(self, cap, clip_idx):
        """
        args:
        =cap: cv2.VideoCapture object

        initializes cv2.VideoWriter object
        """
        # HACK: divide by 2
        fps = cap.get(cv2.CAP_PROP_FPS) / 2
        fourcc = cv2.VideoWriter_fourcc(*"DIB ")
        dims = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(filename=self.out_path + str(clip_idx) + '.mp4',
                              fourcc=fourcc,
                              fps=fps,
                              frameSize=dims,
                              apiPreference=int(cv2.CAP_FFMPEG))

        return(out)


    def _reread_frame(self, cap):
        """
        args:
        =cap: cv2.VideoCapture object

        waits and rereads a stuck frame
        """
        last_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx - 1)
        cv2.waitKey(1000)

        return(cap.read())


    def _remove_clip(self, clip_idx):
        """
        args:
        =clip_idx: zero-indexed clip ID

        removes clips that are too short
        """
        os.remove(self.out_path + str(clip_idx) + '.avi')


    def trim(self, avi_path):
        """
        args:
        =avi_path: path of the .avi file, duh
        """
        cap = cv2.VideoCapture(avi_path)

        while not cap.isOpened():
            cap = cv2.VideoCapture(avi_path)
            cv2.waitKey(1000)

        clip_idx = 0
        while True:
            clip_idx += 1
            frame_idx = 0
            out = self._init_writer(cap, clip_idx)
            __, old_frame = cap.read()
            flag, frame = cap.read()

            while self._difference(old_frame, frame) > self.threshold:
                if not flag:
                    flag, frame = self._reread_frame(cap)

                if frame_idx == 1:
                    out.write(old_frame)
                    out.write(frame)
                else:
                    out.write(frame)

                # HACK: divide by 2
                if np.isclose(cap.get(cv2.CAP_PROP_POS_FRAMES),
                              cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2,
                              atol=0.51, rtol=0.0):
                    os._exit(0)

                frame_idx += 1
                old_frame = frame
                flag, frame = cap.read()

                ### DEBUG
                print(clip_idx + 1, cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_idx < self.frame_seq - 1:
                self._remove_clip(clip_idx)
                clip_idx -= 1


