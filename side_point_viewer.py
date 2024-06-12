import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import OrderedDict


# Paths for the Side and Rear JSON files and image directories
DATA_PATH = "C:/Users/user/Desktop/dataset/archive/www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/"

JSON_PATH_SIDE = os.path.join(DATA_PATH, "Vector/B4/Side/data/coco_b4_side.json")
#IMAGE_DIR_SIDE = os.path.join(DATA_PATH, "Vector/B4/Side/data/images")
file_name_dir = Path(DATA_PATH,"Vector/B4/Side/data/images")

# 파일 이름에서 숫자 추출 및 정렬을 위한 키 생성 함수
def extract_numbers(file_name):
    numbers = re.findall(r'\d+', file_name.name)
    # 파일 이름에 숫자가 여러 개 있을 경우를 대비하여 모든 숫자를 정수로 변환하여 튜플로 반환합니다.
    # 숫자가 없는 경우 파일 이름 정렬에 영향을 주지 않도록 (0,)을 반환합니다.
    return tuple(map(int, numbers)) if numbers else (0,)

visualization = False
Print_info = False

# 디렉토리 내의 모든 파일 경로를 리스트로 가져오고, extract_numbers 함수를 사용하여 정렬
sorted_filenames = sorted(file_name_dir.iterdir(), key=extract_numbers)


def visualize_images(json_path, sorted_filenames, keypoint_names, visualization=False):
    with open(json_path, 'r') as file:
        coco_data = json.load(file)
    
    images_info = {Path(img['file_name']).name: img['id'] for img in coco_data['images']}
    keypoints_info = {anno['image_id']: anno['keypoints'] for anno in coco_data['annotations']}

    k = 1
    distance_data = OrderedDict()

    for file_path in sorted_filenames:
        image_filename = file_path.name
        if image_filename in images_info:
            image_id = images_info[image_filename]
            if image_id in keypoints_info:
                keypoint_data = keypoints_info[image_id]
                keypoint_x_vals = keypoint_data[0::3]
                keypoint_y_vals = keypoint_data[1::3]

                image = Image.open(file_path)
                if visualization == True:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)

                distances = []

                # 모든 키포인트 사이의 거리 계산
                for i in range(len(keypoint_names)):
                    for j in range(i+1, len(keypoint_names)):
                        distance = np.sqrt(
                            (keypoint_x_vals[i] - keypoint_x_vals[j])**2 + 
                            (keypoint_y_vals[i] - keypoint_y_vals[j])**2
                        )

                        # 시각화 옵션이 활성화되어 있으면 선 그리기
                        if visualization == True:
                            plt.plot([keypoint_x_vals[i], keypoint_x_vals[j]], 
                                     [keypoint_y_vals[i], keypoint_y_vals[j]], 'b-')
                            plt.scatter([keypoint_x_vals[i], keypoint_x_vals[j]], 
                                        [keypoint_y_vals[i], keypoint_y_vals[j]], c='r', marker='x')
                            plt.text(keypoint_x_vals[i], keypoint_y_vals[i], keypoint_names[i], color='white')
                            plt.text(keypoint_x_vals[j], keypoint_y_vals[j], keypoint_names[j], color='white')
                        
                        # 거리 데이터 리스트에 추가
                        distances.append(distance)

                distance_data[image_filename] = distances

                if visualization == True:
                    plt.title(f"Image with Keypoints from {json_path.split('/')[-2]}")
                    plt.show()

            k += 1

    # 결과를 JSON 파일로 저장
    with open('all_distance_data_side.json', 'w') as file:
        json.dump(distance_data, file, ensure_ascii=False, indent=4)


side_keypoint_names = ["1_wither", "2_pinbone", "3_shoulderbone", "5_front_girth_bottom",
                        "4_front_girth_top", "9_Height_bottom", "8_Height_top",
                        "7_rear_girth_bottom", "6_rear_girth_top"]

visualize_images(JSON_PATH_SIDE, sorted_filenames, side_keypoint_names)

