# import pandas as pd
# import mysql.connector
# from sqlalchemy import create_engine

# # Load the CSV file
# file_path = './final_merged_data.csv'
# data = pd.read_csv(file_path)

# # Database connection details
# db_user = 'root'
# db_password = 'wngur9836'
# db_host = 'localhost'
# db_port = '3306'  # Default MySQL port
# db_name = 'recommendation_data_db'

# # Establish connection to MySQL using SQLAlchemy
# engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# # Save the data to the database, replacing the table if it exists
# data.to_sql('recommendation', con=engine, if_exists='replace', index=False)

# print("Data saved to the database successfully.")

import pandas as pd

# Define the data in Korean and include the X_COORD and Y_COORD columns
data = {
    "TRAVEL_ID": ["c012001", "c012002", "c012003", "c012004", "c012005", "c012006", "c012007", "c012008", "c012009", "c012010"],
    "TRAVELER_ID": ["24;30;28;", "25;29;32;", "22;23;24;", "35;36;37;", "29;30;33;", "28;29;30;", "31;32;34;", "26;27;28;", "33;34;35;", "38;39;40;"],
    "TRAVEL_PURPOSE": ["가족 여행", "혼자 모험", "친구 여행", "문화 탐방", "출장", "친구와 로드트립", "단기 여행", "등산 여행", "자연 휴양", "비즈니스 컨퍼런스"],
    "MVMN_NM": ["자가용", "대중교통", "자전거", "도보", "버스", "자가용", "기차", "대중교통", "자가용", "비행기"],
    "AGE_GRP": [30, 25, 22, 35, 29, 28, 31, 26, 33, 38],
    "GENDER": ["여성", "남성", "여성", "남성", "여성", "남성", "여성", "남성", "여성", "남성"],
    "TRAVEL_STYL_1": [5, 4, 3, 6, 4, 7, 5, 4, 6, 5],
    "TRAVEL_MOTIVE_1": [1.0, 2.0, 1.5, 3.0, 1.0, 2.5, 1.2, 1.8, 2.0, 3.5],
    "TRAVEL_STATUS_ACCOMPANY": [2, 3, 2, 1, 2, 3, 1, 2, 3, 1],
    "TRAVEL_STATUS_DAYS": [2, 3, 2, 1, 2, 3, 1, 2, 3, 1],
    "VISIT_AREA_NM": ["대천해수욕장", "해운대해수욕장", "남이섬", "경복궁", "인천국제공항", "설악산 국립공원", "전주 한옥마을", "한라산", "주왕산 국립공원", "코엑스 컨벤션센터"],
    "ROAD_ADDR": ["충청남도 보령시", "부산광역시", "강원도 춘천시", "서울특별시", "인천광역시", "강원도 속초시", "전라북도 전주시", "제주특별자치도", "경상북도 청송군", "서울특별시"],
    "X_COORD": [126.550231, 129.158981, 127.525217, 126.976857, 126.450516, 128.465911, 127.1481, 126.542343, 129.170384, 127.059447],
    "Y_COORD": [36.292322, 35.158866, 37.791618, 37.579617, 37.460234, 38.119441, 35.811976, 33.361667, 36.416925, 37.511196]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('./new_data.csv', index=False)

print("CSV file 'new_data.csv' created successfully!")
