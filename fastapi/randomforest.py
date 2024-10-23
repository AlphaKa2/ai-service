import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, dtype={'TRAVEL_ID': str}, low_memory=False)
    data['MVMN_NM'] = data['MVMN_NM'].ffill()  # Forward fill missing values in MVMN_NM
    return data

# Handle encoding and scaling
def encode_and_scale_data(data):
    label_encoders = {}
    categorical_cols = ['TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER', 'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR', 'VISIT_AREA_NM']
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    return data, label_encoders

# Split the data into train/test sets
def split_data(data):
    X = data.drop(columns=['TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'X_COORD', 'Y_COORD'])
    y = data['TRAVELER_ID'].values  # Keep TRAVELER_ID for test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the Random Forest model
def train_rf_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Generate travel recommendations using Random Forest based on new input data
def get_travel_recommendations(rf_model, new_input_data, data, label_encoders):
    # Define the expected column order (order must match what the model was trained on)
    expected_columns_order = [
        'TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER',
        'TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1',
        'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_STATUS_DAYS', 'ROAD_ADDR'
    ]

    # Encode categorical input data
    new_input_encoded = {}
    categorical_cols = ['TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER', 'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR']
    
    for col in categorical_cols:
        new_input_encoded[col] = label_encoders[col].transform([new_input_data[col]])[0]
    
    # Handle integer columns directly without encoding
    int_cols = ['TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1', 'TRAVEL_STATUS_DAYS']
    for col in int_cols:
        new_input_encoded[col] = new_input_data[col]

    # Ensure that the encoded input data is ordered correctly according to expected_columns_order
    input_array = np.array([new_input_encoded[col] for col in expected_columns_order]).reshape(1, -1)

    # Use the Random Forest model to predict the top travel recommendations
    predicted_travelers = rf_model.predict(input_array)

    # Decode the road address back to the original value using LabelEncoder
    test_road_addr_encoded = new_input_encoded['ROAD_ADDR']
    test_road_addr = label_encoders['ROAD_ADDR'].inverse_transform([test_road_addr_encoded])[0]

    # Extract the corresponding rows from the data for matching destinations
    matching_destinations = data[data['TRAVELER_ID'].isin(predicted_travelers)]

    # Decode categorical columns for the recommendations
    decoded_destinations = []
    for _, row in matching_destinations.iterrows():
        matched_destination = {
            "TRAVEL_PURPOSE": label_encoders['TRAVEL_PURPOSE'].inverse_transform([int(row['TRAVEL_PURPOSE'])])[0],
            "MVMN_NM": label_encoders['MVMN_NM'].inverse_transform([int(row['MVMN_NM'])])[0],
            "AGE_GRP": label_encoders['AGE_GRP'].inverse_transform([int(row['AGE_GRP'])])[0],
            "GENDER": label_encoders['GENDER'].inverse_transform([int(row['GENDER'])])[0],
            "TRAVEL_STYL_1": int(row['TRAVEL_STYL_1']),
            "TRAVEL_MOTIVE_1": int(row['TRAVEL_MOTIVE_1']),
            "TRAVEL_STATUS_ACCOMPANY": label_encoders['TRAVEL_STATUS_ACCOMPANY'].inverse_transform([int(row['TRAVEL_STATUS_ACCOMPANY'])])[0],
            "TRAVEL_STATUS_DAYS": int(row['TRAVEL_STATUS_DAYS']),
            "ROAD_ADDR": label_encoders['ROAD_ADDR'].inverse_transform([int(row['ROAD_ADDR'])])[0],
            "TRAVELER_ID": row['TRAVELER_ID']
        }
        decoded_destinations.append(matched_destination)

    return decoded_destinations, test_road_addr, new_input_encoded, input_array

# Calculate recall based on user visited places
def calculate_recall(matching_destinations, y_test):
    recommended_travel_ids = [dest['TRAVELER_ID'] for dest in matching_destinations]
    visited_set = set(y_test)  # The user's actual visited places from test data
    recommended_set = set(recommended_travel_ids)

    # Find how many visited places were correctly recommended
    correctly_recommended = visited_set.intersection(recommended_set)

    recall = len(correctly_recommended) / len(visited_set) if len(visited_set) > 0 else 0
    return recall

# Save matching destinations and input data to JSON
def save_to_json(matching_destinations, input_data, file_path="output.json"):
    # Ensure all NumPy types are converted to Python native types for JSON serialization
    input_data = input_data.astype(float).tolist()  # Convert numpy array to float and then to list

    # Convert all int64 to int and float64 to float in matching_destinations
    matching_destinations = [
        {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
         for k, v in destination.items()} for destination in matching_destinations
    ]
    output_data = {
        "input_data": input_data,  # Already converted to list
        "matching_destinations": matching_destinations
    }

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    file_path = './final_merged_data.csv'
    data = load_and_preprocess_data(file_path)
    data, label_encoders = encode_and_scale_data(data)
    X_train, X_test, y_train, y_test = split_data(data)  # y_test contains the actual TRAVELER_IDs
    rf_model = train_rf_model(X_train, y_train)

    # Example new input data
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

    matching_destinations, test_road_addr, new_input_encoded, input_array = get_travel_recommendations(
        rf_model, new_input_data, data, label_encoders
    )

    # Calculate recall using test data (y_test)
    recall = calculate_recall(matching_destinations, y_test)
    print(f"Recall: {recall}")

    # Save matching destinations and input data to JSON
    save_to_json(matching_destinations, input_array)
