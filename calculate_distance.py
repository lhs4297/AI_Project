import json
import re
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError


import matplotlib.pyplot as plt

def open_distance_data():
    true_weight_list = []
    weight_list = []
    weight_adj_list = []
    # 파일 열기
    with open('distance_data_rear.json', 'r') as ddr:
        dict_data_rear = json.load(ddr)
    with open('distance_data_side.json', 'r') as dds:
        dict_data_side = json.load(dds)
    with open('diameter_data.json', 'r') as isi:
        diameter_data = json.load(isi)

    # 파일 이름에서 공통 부분을 기준으로 매칭
    for file_name_d, diameter in diameter_data.items():
        extracted_part_d = '_'.join(file_name_d.split('_')[:2]) + '_'
        
        # 숫자 추출
        match = re.search(r'_([0-9]+)_', file_name_d)
        if match:
            true_weight = match.group(1)  # 매치된 숫자 부분

        for file_name_rear, distances in dict_data_rear.items():
            extracted_part_rear = '_'.join(file_name_rear.split('_')[:2]) + '_'
            if extracted_part_d == extracted_part_rear:
                c_distances_rear = distances["C_distance_1_to_2"]
                d_distances_rear = distances["D_distance_3_to_4"]

                for file_name_side, distances in dict_data_side.items():
                    extracted_part_side = '_'.join(file_name_side.split('_')[:2]) + '_'
                    if extracted_part_d == extracted_part_side:
                        a_distances_side = distances["A_distance_2_to_3"]
                        b_distances_side = distances["B_distance_5_to_4"]
                        c_distances_side = distances["C_distance_6_to_7"]

                        # 실제 거리 계산
                        true_distance_c_side = 3.93701*c_distances_side / diameter
                        true_distance_a_side = a_distances_side * true_distance_c_side / c_distances_side
                        true_distance_b_side = b_distances_side * true_distance_c_side / c_distances_side

                        true_distance_d_rear = d_distances_rear * true_distance_c_side / c_distances_rear

                        # 무게 계산
                        weight = true_distance_a_side * true_distance_b_side * true_distance_b_side / 300 / 2.205 # kg로 변환
                        adj = math.log(weight)**2 / 100 + 1
                        weight_adj = weight * adj 
                        weight_2 = true_distance_a_side * true_distance_b_side * true_distance_d_rear / 300 

                        # 결과 출력
                        Error = weight -int(true_weight)
                        Error_adj = weight_adj -int(true_weight)

                        #리스트에 담기
                        true_weight_list.append(int(true_weight))
                        weight_list.append(weight)
                        weight_adj_list.append(weight_adj)

                        # print("----------------------------------------------")
                        # # print(f"실제무게:{true_weight}")
                        # # print(math.tan(diameter))
                        # # print(math.log10(diameter))
                        # # print(math.log(weight))

                        # print(f"파일명: {file_name_d}")
                        # print(f"A거리의 실제 거리:{true_distance_a_side:.2f}cm")
                        # print(f"B거리의 실제 거리:{true_distance_b_side:.2f}cm")
                        # print(f"C거리의 실제 거리:{true_distance_c_side:.2f}cm")
                        # print(f"측정무게: {weight:.2f}kg | 실제무게 : {true_weight} | Error : {Error:.2f}" )
                        # print(f"보정값:{adj:.2f}")
                        # print(f"보정된무게:{weight_adj:.2f} | Error : {Error_adj:.2f}")
                        break  # 일치하는 첫번째 side 파일에 대해 계산을 완료했으므로 반복 종료

    return true_weight_list, weight_list, weight_adj_list

def plot_data(true_weight_list, weight_list, weight_adj_list):
    # 인덱스 리스트 생성
    indices = list(range(len(true_weight_list)))
    
    # true_weight 값에 대한 plot
    plt.plot(indices, true_weight_list, label='True Weight', marker='o')
    
    # weight 값에 대한 plot
    plt.plot(indices, weight_list, label='Weight', marker='x')
    
    # weight_adj 값에 대한 plot
    plt.plot(indices, weight_adj_list, label='Weight Adjusted', marker='*')
    
    # 레이블 및 범례 추가
    plt.xlabel('Index')
    plt.ylabel('Weight')
    plt.title('Weight Comparison')
    plt.legend()
    
    # 그래프 표시
    plt.show()

def plot_data_model(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    
    # 실제 무게
    plt.plot(range(len(y_test)), y_test, color='blue', label='true_weight', alpha=0.5, marker='*')
    
    # 예측된 무게
    plt.plot(range(len(y_pred)), y_pred, color='red', label='trained_weight', alpha=0.5, marker='x')
    
    plt.title('true weight vs trained weight')
    plt.xlabel('sample_index')
    plt.ylabel('weight')
    plt.legend()
    plt.show()
# Adam

# 데이터 로드
true_weight_list, weight_list, weight_adj_list = open_distance_data()
X = np.array(weight_adj_list).reshape(-1, 1)
y = np.array(true_weight_list)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.01)

# 모델 컴파일
model.compile(optimizer=optimizer, loss=MeanAbsoluteError())

# 모델 학습
model.fit(X_train, y_train, epochs=1000, verbose=1)

# 예측 및 평가
y_pred = model.predict(X_test)

# MSE 계산 (이 부분은 직접 구현해야 합니다.)
# 예를 들어, sklearn의 mean_squared_error를 사용할 수 있습니다.
mse = mean_squared_error(y_test, y_pred.flatten())
mae = mean_absolute_error(y_test, y_pred.flatten())
print("평균 절대 오차(MAE):", mae)
print("평균 제곱 오차(MSE):", mse)
plot_data_model(y_test, y_pred)