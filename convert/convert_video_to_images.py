import argparse
import os

import cv2
from tqdm import tqdm
from time import sleep


def _tqdm_test(length, video):
    with tqdm(total=length) as pbar:
        while True:
            length -= 1
            pbar.update(1)
            sleep(0.001)
            if length == 0:
                break

def _video_info(video):
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return width, height, frames, fps


def convert_to_images(video_path, save_interval=1, out_path=None):
    video  = cv2.VideoCapture(video_path)

    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(video.get(cv2.CAP_PROP_FPS))

    outpath = os.path.join(video_path[:-4], 'images')
    if out_path:
        outpath = os.path.join(out_path, os.path.basename(video_path)[:-4])

    converting_frames = int(frames / save_interval)
    print(f'Total Frames: {frames}')
    print(f'Width: {width}, Height: {height}')
    print(f'Converting Frames: {converting_frames}')

    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath, exist_ok=True)
    except OSError:
        print(f'Error: Creating directory. {outpath}')

    success = True
    # fps = save_interval

    with tqdm(total=converting_frames) as pbar:
        while success:
            success, image = video.read()
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    
            if ( current_frame % save_interval == 0 and success):
                fname = f'{outpath}/{current_frame:06d}.jpg'
                cv2.imwrite(fname, image)

                pbar.update(1)

    print('Finish! convert video to frame')

    video.release()


def _parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=1)

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':    
    cfg = _parse_config()
    assert cfg.input, 'Error input is None'
    assert cfg.output, 'Error output is None'

    convert_to_images(video_path=cfg.input, out_path=cfg.output, save_interval=cfg.save_interval)