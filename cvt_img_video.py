import argparse
import glob

import cv2


def generate_video_from_image(images, output, fps=30):
    frames = []
    for idx , image in enumerate(images) : 
        if (idx % 2 == 0) | (idx % 5 == 0) :
            continue
        img = cv2.imread(image)
        height, width, _ = img.shape
        size = (width, height)
        frames.append(img)

    video_writer = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):

        video_writer.write(frames[i])
    video_writer.release()


def convert_images_to_video(img_path, output, fps=30):
    reg_text = f'{img_path}/*.jpg'
    images = glob.glob(reg_text)
    images = sorted(images)

    generate_video_from_image(images, output, fps)


def main(args):
    convert_images_to_video(args.img_dir, args.output, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default=None, help='image path')
    parser.add_argument('--output', type=str, default=None, help='image path')
    parser.add_argument('--fps', type=float, default=30, help='image path')

    args = parser.parse_args()

    main(args)
