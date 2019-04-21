import os
import cv2
import numpy as np


VID_DIR = "input_video"
IN_DIR = "output_images"
OUT_DIR = "output"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def video_generator(num):
    out_path = "output/marked_video1.mp4"
    video_out = mp4_video_writer(out_path, (1080, 1920), 20)        
    for i in range(1, num+1):
        image_name = "marked_{}.png".format(i)
        image = cv2.imread(os.path.join(IN_DIR, image_name))
        video_out.write(image)
    video_out.release()
    
def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(IN_DIR, filename), image)


if __name__ == '__main__':
    video_generator(47)
