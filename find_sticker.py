import cv2
import numpy as np
import os
import json
from PIL import Image
from pathlib import Path
import re

# 이미지가 있는 디렉토리 경로
directory_path = Path("C:/Users/user/Desktop/dataset/archive/www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/Pixel/B4/Side/annotations")

# 이미지 정보를 저장할 리스트 초기화
images_info = []

# 이미지 ID 초기화
image_id = 1

# 파일 이름에서 숫자 추출 및 정렬을 위한 키 생성 함수
def extract_numbers(file_name):
    numbers = re.findall(r'\d+', file_name.name)
    # 파일 이름에 숫자가 여러 개 있을 경우를 대비하여 모든 숫자를 정수로 변환하여 튜플로 반환합니다.
    # 숫자가 없는 경우 파일 이름 정렬에 영향을 주지 않도록 (0,)을 반환합니다.
    return tuple(map(int, numbers)) if numbers else (0,)

# 디렉토리 내의 모든 파일 경로를 리스트로 가져오고, extract_numbers 함수를 사용하여 정렬
sorted_filenames = sorted(directory_path.iterdir(), key=extract_numbers)

# 디렉토리 순회
for filename in  sorted_filenames:
    if filename.suffix in (".png", ".jpg"):  # JPG 이미지만 처리
        # 이미지 파일의 전체 경로
        file_path = directory_path / filename
        
        # 이미지 읽기
        image = cv2.imread(str(file_path.absolute()))

        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 파란색 스티커의 HSV 범위
        blue_lower = np.array([100, 150, 0], np.uint8)
        blue_upper = np.array([140, 255, 255], np.uint8)

        # 파란색 범위 내의 객체만 분리하기 위한 마스크 생성
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # 마스크에서 노이즈 제거
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 파란색 객체의 경계 찾기
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sticker_info = {}
        if contours:
            contour = max(contours, key=cv2.contourArea)
            left = tuple(contour[contour[:, :, 0].argmin()][0])
            right = tuple(contour[contour[:, :, 0].argmax()][0])
            top = tuple(contour[contour[:, :, 1].argmin()][0])
            bottom = tuple(contour[contour[:, :, 1].argmax()][0])

            # 스티커 좌표 정보 딕셔너리에 저장
            sticker_info = {
                "left": [int(i) for i in left],
                "right": [int(i) for i in right],
                "top": [int(i) for i in top],
                "bottom": [int(i) for i in bottom]
            }

        # 이미지 크기 가져오기
        with Image.open(file_path) as img:
            width, height = img.size
        
        # 이미지 정보를 딕셔너리로 저장
        image_info = {
            "id": image_id,
            "file_name": filename.name,
            "sticker_info": sticker_info
        }
        
        # 리스트에 추가
        images_info.append(image_info)
        
        # 이미지 ID 업데이트
        image_id += 1
        print(f"image_id{image_id} 저장중...")

# 전체 정보를 JSON으로 저장
json_data = {"images": images_info}
with open("images_sticker_info.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print("이미지와 스티커 정보가 images_sticker_info.json 파일에 저장되었습니다.")

