import os
import numpy as np
import imageio


def frame_average(input_path, output_path, num_frames, step_size):
    """
    args:
    =input_path: path of input frames, named consecutively
    =ouput_path: path to output two directories:
        - blur: for averaged frames
        - sharp: for middle frames
    =num_frames: number of frames to average
    =step_size: frames to skip between ground truths
    """
    last_frame_idx = max([int(f.replace('.png', '')) for f in os.listdir(input_path) 
                          if os.path.isfile(os.path.join(input_path, f))])
    k = int(num_frames / 2)
    sharp_frames_idxs = np.arange(k + 1, last_frame_idx + 1, step_size)
    blur_outpath = os.path.join(output_path, 'blur/')
    sharp_outpath = os.path.join(output_path, 'sharp/')

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(blur_outpath):
        os.mkdir(blur_outpath)
    if not os.path.exists(sharp_outpath):
        os.mkdir(sharp_outpath)

    for count_idx, example_idx in enumerate(sharp_frames_idxs):
        blur_frame = np.mean(np.concatenate(
                        [np.expand_dims(
                         imageio.imread(
                         os.path.join(input_path, str(int(frame_idx)) + '.png')), 
                         axis=0)
                         for frame_idx in np.arange(example_idx-k, example_idx+k+1, 1)],
                        axis=0), axis=0)
        sharp_frame = imageio.imread(os.path.join(input_path,
                                     str(int(example_idx)) + '.png'))
        blur_img_outpath = os.path.join(blur_outpath, str(count_idx) + '.png')
        sharp_img_outpath = os.path.join(sharp_outpath, str(count_idx) + '.png')        
        imageio.imsave(blur_img_outpath, blur_frame.astype(dtype=np.uint8))
        imageio.imsave(sharp_img_outpath, sharp_frame.astype(dtype=np.uint8))

