import os
import glob
import random

def find_image_files(root_directory, output_file):
    # 지원하는 이미지 파일 확장자 목록
    image_extensions = ['.jpg', '.jpeg', '.png']

    # 모든 이미지 파일 경로를 저장할 리스트
    image_files = []

    # 모든 파일 경로를 가져오기
    for extension in image_extensions:
        image_files.extend(glob.glob(root_directory + '/**/*' + extension, recursive=True))

    # 파일 목록을 임의로 섞기
    for i in range(random.randint(20, 30)):
        random.shuffle(image_files)

    # 파일에 저장
    with open(output_file, 'w') as f:
        for image_file in image_files:
            f.write(image_file + '\n')

    print(f'Found and saved {len(image_files)} image files.')


if __name__ == '__main__':
    # 테스트할 루트 디렉토리 경로와 출력 파일 경로 설정
    root_directory_path = '/mnt/hdd01/hyeongdo/datasets/VP-SAR-v1.0.0.all/Test'
    output_file_path = '/mnt/hdd01/hyeongdo/datasets/VP-SAR-v1.0.0.all/test.txt'

    find_image_files(root_directory_path, output_file_path)