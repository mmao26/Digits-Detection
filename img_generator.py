import os
import cv2
import numpy as np


VID_DIR = "input_video"
IN_DIR = "input_images"

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None

def video_process(video_name, ratio):
    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.next()

    frame_num = 1
    nn = 1
    while image is not None:
        print ("Processing fame {}".format(frame_num))
        if frame_num % ratio == 0:
            filename1 = "orig_{}.png".format(nn)
            cv2.imwrite(os.path.join(IN_DIR, filename1), image)
            print (image.shape)
            nn += 1
        image = image_gen.next()
        frame_num += 1

if __name__ == '__main__':
    video_process("video1.mp4", 5)
