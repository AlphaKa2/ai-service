from typing import Dict, List

# Define the mapping for means of transportation conversion (MVMN_NM)
MVMN_NM_MAP: Dict[str, str] = {
    'CAR': '자가용',
    'PUBLIC_TRANSPORTATION': '대중교통 등'
}
# Define the mapping for TRAVEL_PURPOSE conversion
TRAVEL_PURPOSE_MAP: Dict[str, str] = {
    'SHOPPING': "1;",
    'THEME_PARK': "2;",
    'HISTORIC_SITE': "3;",
    'CITY_TOUR': "4;",
    'OUTDOOR_SPORTS': "5;",
    'CULTURAL_EVENT': "6;",
    'NIGHTLIFE': "7;",
    'CAMPING': "8;",
    'LOCAL_FESTIVAL': "9;",
    'SPA': "10;",
    'EDUCATION': "11;",
    'FILM_LOCATION': "12;",
    'PILGRIMAGE': "13;",
    'WELLNESS': "21;",
    'SNS_SHOT': "22;",
    'HOTEL_STAYCATION': "23;",
    'NEW_TRAVEL_DESTINATION': "24;",
    'PET_FRIENDLY': "25;",
    'INFLUENCER_FOLLOW': "26;",
    'ECO_TRAVEL': "27;",
    'HIKING': "28;"
}

GENDER_MAP: Dict[str, str] = {
    '남': 'MALE',
    '여': 'FEMALE'
}

GENDER_REVERSE_MAP: Dict[str, str] = {
    'MALE': '남',
    'FEMALE': '여'
}

# Define the mapping for style conversion
TRAVEL_STYL_1_MAP: Dict[str, int] = {
    'VERY_NATURE': 1,
    'MODERATE_NATURE': 2,
    'NEUTRAL': 3,
    'MODERATE_CITY': 4,
    'VERY_CITY': 5
}

# Define the mapping for motive conversion
TRAVEL_MOTIVE_1: Dict[str, int] = {
    'ESCAPE': 1,
    'REST': 2,
    'COMPANION_BONDING': 3,
    'SELF_REFLECTION': 4,
    'SOCIAL_MEDIA': 5,
    'EXERCISE': 6,
    'NEW_EXPERIENCE': 7,
    'CULTURAL_EDUCATION': 8,
    'SPECIAL_PURPOSE': 9
}

# Define the mappings
AGE_GRP_MAP: Dict[str, str] = {
    'UNDER_9': "10",
    'TEENS': "10",
    '20S': "20",
    '30S': "30",
    '40S': "40",
    '50S': "50",
    '60S': "60",
    '70_AND_OVER': "70"
}

# Define the mapping for travel status accompany (TRAVEL_STATUS_ACCOMPANY)
TRAVEL_STATUS_ACCOMPANY_MAP: Dict[str, str] = {
    'GROUP_OVER_3': '3인 이상 여행(가족 외)',
    'WITH_CHILD': '자녀 동반 여행',
    'DUO': '2인 여행(가족 외)',
    'SOLO': '나홀로 여행',
    'FAMILY_DUO': '2인 가족 여행',
    'EXTENDED_FAMILY': '3대 동반 여행(친척 포함)'
}

TRANSPORTATION_MAP: Dict[str, str] = {
    'CAR': '자가용',
    'PUBLIC_TRANSPORTATION': '대중교통 등'
}