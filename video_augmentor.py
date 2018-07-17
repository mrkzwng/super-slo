import numpy as np 
import cv2
import os
from scipy.misc import imread, imsave, imshow, imresize, imsave


class Video_Augmentor:
    def __init__(self, out_path, shape, frame_seq=3,
                       crop=True, zoom=2, reflect='both'):
        """
        args:
        =out_path: output path of the .jpg frames and 
            .txt file of file names
        =shape: output shape of the .jpg frames
        =frame_seq: number of frames per sequence
        =crop: whether to random crop
        =
        """
        self.crop = crop
        self.zoom = zoom
        self.reflect = reflect
        self.out_path = out_path

    def _crop(self, imgs):
        """
        args:
        =imgs: frame_seq X height X width X 3 np.ndarray 
        """


    def _zoom(self, imgs):
        """
        args:
        =imgs: frame_seq X height X width X 3 np.ndarray 
        """


    def _reflect(self, imgs):
        """
        args:
        =imgs: frame_seq X height X width X 3 np.ndarray 
        """


    def _save(self, imgs):
        """
        args:
        =imgs: frame_seq X height X width X 3 np.ndarray 
        """


    def augment(self, avi_path):
        """
        args:
        =avi_path: input path of the .avi file
        """
