from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from recommendation import app, get_db
import logging
from datetime import date, datetime

client = TestClient(app)

logging.basicConfig(level=logging.info)  # or INFO, WARNING, etc.
logger = logging.getLogger(__name__)

@patch("recommendation.get_db")
def test_recommendation_trip_not_found(mock_get_db):
    """
    Test Case 1: Trip does not exist (404 response)
    """
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    # Mock the query to return None
    mock_db.query.return_value.filter_by.return_value.first.return_value = None

    response = client.get("/recommendations/1", headers={"X-User-Id": "123"})
    assert response.status_code == 404
    assert response.json() == {"detail": "존재하지 않는 여행입니다."}


@patch("recommendation.SessionLocal")  # Mock the `SessionLocal` dependency
def test_recommendation_trip_exists(mock_session_local):
    """
    Test Case: Trip exists and the user has permission (200 response)
    """
    # Create a mock for the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock RecommendationPlan data
    mock_plan = MagicMock()
    mock_plan.recommendation_trip_id = 1
    mock_plan.user_id = 1  # Ensure this matches the test header
    mock_plan.name = "Test Trip"
    mock_plan.description = "Test Description"
    mock_plan.recommendation_type = "AI-GENERATED"
    mock_plan.start_date = date(2024, 1, 1)  # Use `datetime.date` for valid `.isoformat()`
    mock_plan.end_date = date(2024, 1, 10)   # Use `datetime.date` for valid `.isoformat()`
    mock_plan.created_at = datetime(2024, 1, 1, 12, 0, 0)  # Use `datetime.datetime`
    mock_plan.deleted_at = None

    # Mock Preference data
    mock_preference = MagicMock()
    mock_preference.preference_id = None  # Explicitly set to Python's None

    # Configure `filter_by` for different queries
    def mock_filter_by(**kwargs):
        if "recommendation_id" in kwargs:
            return MagicMock(first=MagicMock(return_value=mock_preference))
        elif "recommendation_trip_id" in kwargs:
            return MagicMock(first=MagicMock(return_value=mock_plan))
        else:
            return MagicMock(first=MagicMock(return_value=None))

    mock_db.query.return_value.filter_by.side_effect = mock_filter_by

    # Mock behavior for `query().filter_by().all()` for RecommendationDay and schedules
    mock_db.query.return_value.filter_by.return_value.all.return_value = []

    # Call the endpoint
    response = client.get("/recommendations/1", headers={"X-User-Id": "1"})

    # Debugging: Print mock database calls
    print("Mock calls to database:")
    print(mock_db.mock_calls)

    # Assert that the database query was called
    try:
        assert mock_db.query.called, "Database query was not called!"
        print("Database query was called successfully.")
    except AssertionError as e:
        print("Database query was NOT called!")
        raise e

    # Assert the response
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert response.json() == {
        "title": "Test Trip",
        "description": "Test Description",
        "recommendation_type": "AI-GENERATED",
        "start_date": "2024-01-01",  # Match the mocked date
        "end_date": "2024-01-10",    # Match the mocked date
        "days": [],  # No days for this test case
        "preference_id": None  # Explicitly set as None for this test case
    }

@patch("recommendation.SessionLocal")  # Mock the `SessionLocal` dependency
def test_recommendation_trip_no_permission(mock_session_local):
    """
    Test Case 3: User has no permission for the trip (403 response)
    """
    # Create a mock for the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock RecommendationPlan data with a different user ID
    mock_plan = MagicMock()
    mock_plan.recommendation_trip_id = 1
    mock_plan.user_id = 999  # Different user ID
    mock_plan.name = "Test Trip"
    mock_plan.description = "Test Description"
    mock_plan.recommendation_type = "AI-GENERATED"
    mock_plan.start_date = date(2024, 1, 1)  # Use `datetime.date` for valid `.isoformat()`
    mock_plan.end_date = date(2024, 1, 10)   # Use `datetime.date` for valid `.isoformat()`
    mock_plan.created_at = datetime(2024, 1, 1, 12, 0, 0)  # Use `datetime.datetime`
    mock_plan.deleted_at = None

    # Configure `filter_by` for RecommendationPlan query
    def mock_filter_by(**kwargs):
        if "recommendation_trip_id" in kwargs and kwargs["recommendation_trip_id"] == "1":  # Ensure matching as a string
            return MagicMock(first=MagicMock(return_value=mock_plan))
        return MagicMock(first=MagicMock(return_value=None))

    mock_db.query.return_value.filter_by.side_effect = mock_filter_by

    # Call the endpoint with a different user ID
    response = client.get("/recommendations/1", headers={"X-User-Id": "123"})  # User ID doesn't match mock_plan.user_id

    # Debugging: Print mock database calls
    print("Mock calls to database:")
    print(mock_db.mock_calls)

    # Assert that the database query was called
    try:
        assert mock_db.query.called, "Database query was not called!"
        print("Database query was called successfully.")
    except AssertionError as e:
        print("Database query was NOT called!")
        raise e
    
    # Assert that the response status code is 403 (Forbidden)
    assert response.status_code == 403, f"Unexpected status code: {response.status_code}"

    # Assert that the error message matches the expected output
    assert response.json() == {"detail": "해당 여행에 대한 권한이 없습니다."}

    # Debugging: Print mock database calls to verify behavior
    print("Mock calls to database:")
    print(mock_db.mock_calls)


@patch("recommendation.SessionLocal")
def test_recommendation_trip_user_permission(mock_session_local):
    """
    Test Case 4: User has permission for the trip (200 response)
    """
    # Create a mock for the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock RecommendationPlan data with a different user ID
    mock_plan = MagicMock()
    mock_plan.recommendation_trip_id = 1
    mock_plan.user_id = 123  # Different user ID
    mock_plan.name = "Test Trip"
    mock_plan.description = "Test Description"
    mock_plan.recommendation_type = "AI-GENERATED"
    mock_plan.start_date = date(2024, 1, 1)  # Use `datetime.date` for valid `.isoformat()`
    mock_plan.end_date = date(2024, 1, 10)   # Use `datetime.date` for valid `.isoformat()`
    mock_plan.created_at = datetime(2024, 1, 1, 12, 0, 0)  # Use `datetime.datetime`
    mock_plan.deleted_at = None

    # Mock Preference data
    mock_preference = MagicMock()
    mock_preference.preference_id = None  # Explicitly set to Python's None

    # Configure `filter_by` for different queries
    def mock_filter_by(**kwargs):
        if "recommendation_id" in kwargs:
            return MagicMock(first=MagicMock(return_value=mock_preference))
        elif "recommendation_trip_id" in kwargs:
            return MagicMock(first=MagicMock(return_value=mock_plan))
        else:
            return MagicMock(first=MagicMock(return_value=None))

    mock_db.query.return_value.filter_by.side_effect = mock_filter_by

    # Mock behavior for `query().filter_by().all()` for RecommendationDay and schedules
    mock_db.query.return_value.filter_by.return_value.all.return_value = []

    response = client.get("/recommendations/1", headers={"X-User-Id": "123"})
    # Debugging: Print mock database calls
    print("Mock calls to database:")
    print(mock_db.mock_calls)

    # Assert that the database query was called
    try:
        assert mock_db.query.called, "Database query was not called!"
        print("Database query was called successfully.")
    except AssertionError as e:
        print("Database query was NOT called!")
        raise e

    # Assert the response
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert response.json() == {
        "title": "Test Trip",
        "description": "Test Description",
        "recommendation_type": "AI-GENERATED",
        "start_date": "2024-01-01",  # Match the mocked date
        "end_date": "2024-01-10",    # Match the mocked date
        "days": [],  # No days for this test case
        "preference_id": None  # Explicitly set as None for this test case
    }
