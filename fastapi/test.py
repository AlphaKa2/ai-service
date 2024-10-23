import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import recall_score

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, dtype={'TRAVEL_ID': str}, low_memory=False)
    data['MVMN_NM'] = data['MVMN_NM'].ffill()  # MVMN_NM의 결측값을 이전 값으로 채움
    return data

# 인코딩 (스케일링은 나중에 수행)
def encode_data(data):
    label_encoders = {}
    categorical_cols = [
        'TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER',
        'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR', 'VISIT_AREA_NM'
    ]
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    return data, label_encoders

# 데이터 분할 (훈련/테스트 세트)
def split_data(data):
    y = data['TRAVELER_ID'].values  # 타겟 변수 추출
    X = data.drop(columns=['TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'X_COORD', 'Y_COORD'], errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# KNN 모델 훈련 (피처 가중치 적용)
def train_knn_model(X_train, feature_weights, expected_columns_order):
    # expected_columns_order에 따라 각 피처의 가중치 정의
    weights = []
    for col in expected_columns_order:
        if col in feature_weights:
            weights.append(feature_weights[col])
        else:
            weights.append(1)  # 기본 가중치
    
    knn_model = NearestNeighbors(
        n_neighbors=20, 
        algorithm='auto', 
        metric='minkowski', 
        p=2, 
        metric_params={'w': weights}
    )
    knn_model.fit(X_train)
    return knn_model

# KNN을 사용하여 여행 추천 생성
def get_travel_recommendations(knn_model, new_input_data, data, label_encoders, scaler, expected_columns_order):
    # 카테고리형 입력 데이터 인코딩
    new_input_encoded = {}
    categorical_cols = [
        'TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER',
        'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR'
    ]
    
    for col in categorical_cols:
        new_input_encoded[col] = label_encoders[col].transform([new_input_data[col]])[0]
    
    # 정수형 피처 처리
    int_cols = ['TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1', 'TRAVEL_STATUS_DAYS']
    for col in int_cols:
        new_input_encoded[col] = new_input_data[col]
    
    # 예상된 피처 순서대로 배열 생성
    input_array = np.array([new_input_encoded[col] for col in expected_columns_order]).reshape(1, -1)
    
    # 동일한 스케일러를 사용하여 입력 데이터 스케일링
    input_array_scaled = scaler.transform(input_array)
    
    # KNN 모델을 사용하여 최근접 이웃 찾기
    distances, indices = knn_model.kneighbors(input_array_scaled)
    
    # 입력 벡터 및 이웃들과의 거리 출력
    print("Input vector (scaled):", input_array_scaled)
    print("Distances to nearest neighbors:", distances)
    
    # 상위 20개의 추천 인덱스 추출
    top_20_indices = indices[0]
    
    # 데이터에서 해당 인덱스의 행 추출 및 디코딩
    matching_destinations = []
    for idx in top_20_indices:
        matched_row = data.iloc[idx][expected_columns_order]  # 실제 X 값 가져오기
        matched_destination = {
            "TRAVEL_PURPOSE": label_encoders['TRAVEL_PURPOSE'].inverse_transform([int(matched_row['TRAVEL_PURPOSE'])])[0],
            "MVMN_NM": label_encoders['MVMN_NM'].inverse_transform([int(matched_row['MVMN_NM'])])[0],
            "AGE_GRP": label_encoders['AGE_GRP'].inverse_transform([int(matched_row['AGE_GRP'])])[0],
            "GENDER": label_encoders['GENDER'].inverse_transform([int(matched_row['GENDER'])])[0],
            "TRAVEL_STYL_1": int(matched_row['TRAVEL_STYL_1']),
            "TRAVEL_MOTIVE_1": int(matched_row['TRAVEL_MOTIVE_1']),
            "TRAVEL_STATUS_ACCOMPANY": label_encoders['TRAVEL_STATUS_ACCOMPANY'].inverse_transform([int(matched_row['TRAVEL_STATUS_ACCOMPANY'])])[0],
            "TRAVEL_STATUS_DAYS": int(matched_row['TRAVEL_STATUS_DAYS']),
            "ROAD_ADDR": label_encoders['ROAD_ADDR'].inverse_transform([int(matched_row['ROAD_ADDR'])])[0],
            "TRAVELER_ID": data.iloc[idx]['TRAVELER_ID']  # Recall 계산을 위한 TRAVELER_ID 포함
        }
        matching_destinations.append(matched_destination)
    return matching_destinations, input_array_scaled

# 사용자가 방문한 장소를 기반으로 Recall 계산
def calculate_recall(matching_destinations, y_test):
    recommended_travel_ids = [dest['TRAVELER_ID'] for dest in matching_destinations]
    visited_set = set(y_test)  # 테스트 데이터에서 사용자가 실제로 방문한 장소
    recommended_set = set(recommended_travel_ids)
    
    # 추천된 장소 중 실제로 방문한 장소 수
    correctly_recommended = visited_set.intersection(recommended_set)
    
    recall = len(correctly_recommended) / len(visited_set) if len(visited_set) > 0 else 0
    return recall

# 매칭된 목적지와 입력 데이터를 JSON으로 저장
def save_to_json(matching_destinations, input_data, file_path="output.json"):
    # JSON 직렬화를 위해 모든 NumPy 타입을 Python 네이티브 타입으로 변환
    input_data = input_data.tolist()  # numpy array를 리스트로 변환
    
    # 매칭된 목적지에서 int64를 int로, float64를 float으로 변환
    matching_destinations = [
        {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
         for k, v in destination.items()} for destination in matching_destinations
    ]
    output_data = {
        "input_data": input_data,  # 이미 리스트로 변환됨
        "matching_destinations": matching_destinations
    }
    
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"{file_path}에 데이터 저장 완료.")

if __name__ == "__main__":
    file_path = './final_merged_data.csv'
    data = load_and_preprocess_data(file_path)
    
    # 피처 가중치 정의
    feature_weights = {
        'TRAVEL_PURPOSE': 2,  # 가중치 2 적용
        'TRAVEL_STYL_1': 2    # 가중치 2 적용
    }
    
    # 카테고리형 데이터 인코딩
    data, label_encoders = encode_data(data)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = split_data(data)  # y_test는 실제 TRAVELER_ID 포함
    
    # 피처 스케일링 (훈련 데이터에 맞춰 스케일러 피팅)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 스케일링된 데이터를 다시 DataFrame으로 변환
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 예상 피처 순서 정의
    expected_columns_order = [
        'TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER',
        'TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1',
        'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_STATUS_DAYS', 'ROAD_ADDR'
    ]
    
    # KNN 모델 훈련
    knn_model = train_knn_model(X_train_scaled, feature_weights, expected_columns_order)
    
    # 예시 입력 데이터
    new_input_data = {
        "TRAVEL_PURPOSE": "21;22;26;",
        "MVMN_NM": "자가용",
        "AGE_GRP": "20",
        "GENDER": "여",
        "TRAVEL_STYL_1": 2,
        "TRAVEL_MOTIVE_1": 1,
        "TRAVEL_STATUS_ACCOMPANY": "자녀 동반 여행",
        "TRAVEL_STATUS_DAYS": 3,
        "ROAD_ADDR": "전남 여수시"
    }
    
    # 추천 생성
    matching_destinations, input_array_scaled = get_travel_recommendations(
        knn_model, new_input_data, data, label_encoders, scaler, expected_columns_order=expected_columns_order
    )
    
    # Recall 계산
    recall = calculate_recall(matching_destinations, y_test)
    print(f"Recall: {recall}")
    
    # JSON으로 저장
    save_to_json(matching_destinations, input_array_scaled, file_path="output.json")
