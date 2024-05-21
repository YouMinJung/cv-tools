import os
import hgtk
import re
import glob

# 한글 자모를 영어로 매핑하는 딕셔너리
jamo_to_eng = {
    'ㄱ': 'g', 'ㄲ': 'kk', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄸ': 'tt', 'ㄹ': 'r', 'ㅁ': 'm', 'ㅂ': 'b', 'ㅃ': 'pp', 'ㅅ': 's', 'ㅆ': 'ss', 'ㅇ': 'ng', 'ㅈ': 'j', 'ㅉ': 'jj', 'ㅊ': 'ch', 'ㅋ': 'k', 'ㅌ': 't', 'ㅍ': 'p', 'ㅎ': 'h',
    'ㅏ': 'a', 'ㅐ': 'ae', 'ㅑ': 'ya', 'ㅒ': 'yae', 'ㅓ': 'eo', 'ㅔ': 'e', 'ㅕ': 'yeo', 'ㅖ': 'ye', 'ㅗ': 'o', 'ㅘ': 'wa', 'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅛ': 'yo', 'ㅜ': 'u', 'ㅝ': 'wo', 'ㅞ': 'we', 'ㅟ': 'wi', 'ㅠ': 'yu', 'ㅡ': 'eu', 'ㅢ': 'ui', 'ㅣ': 'i',
    'ㄺ': 'ln',
}

def hangul_to_english(text):
    # 한글을 자모로 분리
    decomposed = hgtk.text.decompose(text)
    
    # 영어로 변환
    result = []
    for char in decomposed:
        if char in jamo_to_eng:
            result.append(jamo_to_eng[char])
        else:
            result.append(char)
    
    # 조합형 문자열로 재조합
    return ''.join(result).replace('ᴥ', '')  # 조합형 문자열의 중간 기호 제거

def rename_files_and_folders(root_directory):
    # 모든 파일과 폴더의 경로를 가져오기
    all_paths = glob.glob(root_directory + '/**', recursive=True)

    # 파일 및 폴더를 처리하는데 하위 폴더부터 처리하기 위해 역순으로 정렬
    all_paths.sort(key=len, reverse=True)
    # print(all_paths)
    for path in all_paths:
        # print(path)
        if re.search('[가-힣]', path):
            dirpath, filename = os.path.split(path)
            new_filename = hangul_to_english(filename)
            src = os.path.join(dirpath, filename)
            dst = os.path.join(dirpath, new_filename)
            os.rename(src, dst)
            print(f'Renamed: {src} -> {dst}')

if __name__ == '__main__':
    # 테스트할 루트 디렉토리 경로 설정
    root_directory_path = '/mnt/hdd01/hyeongdo/datasets/VP-SAR-v1.0.0.all/Test'
    rename_files_and_folders(root_directory_path)