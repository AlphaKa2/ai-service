from unittest.mock import patch, MagicMock
from datetime import date, datetime
from fastapi.testclient import TestClient
from recommendation import app

client = TestClient(app)

@patch("recommendation.SessionLocal")  # Mock the `SessionLocal` dependency
def test_delete_recommendation_exists(mock_session_local):
    """
    Test Case: Successfully delete an existing recommendation (200 response)
    """
    # Create a mock for the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock RecommendationPlan data
    mock_plan = MagicMock()
    mock_plan.recommendation_trip_id = 1
    mock_plan.user_id = 123  # Same user ID
    mock_plan.name = "Test Trip"
    mock_plan.description = "Test Description"
    mock_plan.start_date = date(2024, 1, 1)
    mock_plan.end_date = date(2024, 1, 10)
    mock_plan.created_at = datetime(2024, 1, 1, 12, 0, 0)

    # Mock `get` to return the mock_plan
    mock_db.query.return_value.get.return_value = mock_plan

    # Call the endpoint
    response = client.delete("/recommendations/1", headers={"X-User-Id": "123"})

    # Assert that the response status code is 200
    assert response.status_code == 200
    assert response.json() == {"data": 1}

    # Assert the delete and commit methods were called
    mock_db.delete.assert_called_once_with(mock_plan)
    mock_db.commit.assert_called_once()

    # Debugging: Print mock database calls
    print("Mock calls to database (Delete Exists):")
    print(mock_db.mock_calls)


@patch("recommendation.get_db")  # Mock the `SessionLocal` dependency
def test_delete_recommendation_not_found(mock_get_db):
    """
    Test Case: Trying to delete a non-existent recommendation (404 response)
    """
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    # Mock the query to return None
    mock_db.query.return_value.filter_by.return_value.first.return_value = None

    response = client.get("/recommendations/1", headers={"X-User-Id": "123"})
    assert response.status_code == 404
    assert response.json() == {"detail": "존재하지 않는 여행입니다."}


@patch("recommendation.SessionLocal")  # Mock the `SessionLocal` dependency
def test_delete_recommendation_no_permission(mock_session_local):
    """
    Test Case: Trying to delete a recommendation without permission (403 response)
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
    mock_plan.start_date = date(2024, 1, 1)
    mock_plan.end_date = date(2024, 1, 10)
    mock_plan.created_at = datetime(2024, 1, 1, 12, 0, 0)
    mock_plan.deleted_at = None

    # Mock `get` to return the mock_plan
    mock_db.query.return_value.get.return_value = mock_plan

    # Call the endpoint with a different user ID
    response = client.delete("/recommendations/1", headers={"X-User-Id": "123"})  # User ID doesn't match mock_plan.user_id

    # Debugging: Print mock database calls
    print("Mock calls to database:")
    print(mock_db.mock_calls)

    # Assert that the database query was called
    assert mock_db.query.called, "Database query was not called!"
    print("Database query was called successfully.")

    # Assert that the response status code is 403 (Forbidden)
    assert response.status_code == 403, f"Unexpected status code: {response.status_code}"

    # Assert that the error message matches the expected output
    assert response.json() == {"detail": "해당 여행에 대한 권한이 없습니다."}





@patch("recommendation.SessionLocal")  # Mock the `SessionLocal` dependency
def test_delete_recommendation_internal_error(mock_session_local):
    """
    Test Case: Internal server error during deletion (500 response)
    """
    # Create a mock for the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock RecommendationPlan data
    mock_plan = MagicMock()
    mock_plan.recommendation_trip_id = 1
    mock_plan.user_id = 123  # Same user ID

    # Mock `get` to return the mock_plan
    mock_db.query.return_value.get.return_value = mock_plan

    # Simulate an error during deletion
    mock_db.commit.side_effect = Exception("Database commit failed")

    # Call the endpoint
    response = client.delete("/recommendations/1", headers={"X-User-Id": "123"})

    # Assert that the response status code is 500
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred while deleting the recommendation: Database commit failed"}

    # Debugging: Print mock database calls
    print("Mock calls to database (Internal Error):")
    print(mock_db.mock_calls)
