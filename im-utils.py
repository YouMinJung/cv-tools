import glob
from PIL import Image
import cv2

def _resize_image(img, ratio=1):
    width, height = img.width * ratio, img.height * ratio
    return img.resize(width, height)


def list_images(path):
    # path에 있는 이미지 파일 목록을 작성한다.
    glob.glob(path)



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

    generate_video_from_image(images, output, fps)

