from sqlalchemy import Column, String, BigInteger, Enum, DateTime, Integer, Date, ForeignKey, DECIMAL, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Table: recommendation_plans (already defined in previous code)
class RecommendationPlan(Base):
    __tablename__ = 'recommendation_plans'

    recommendation_trip_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(String(200), nullable=False)
    recommendation_type = Column(Enum('AI-GENERATED', 'YOUTUBER_FOLLOW'), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

# Table: recommended_days
class RecommendedDay(Base):
    __tablename__ = 'recommended_days'

    day_id = Column(BigInteger, primary_key=True, autoincrement=True)
    recommended_trip_id = Column(BigInteger, ForeignKey('recommendation_plans.recommendation_trip_id', ondelete="CASCADE"), nullable=False)
    day_number = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)

    __table_args__ = (UniqueConstraint('recommended_trip_id', 'day_number', name='_trip_day_uc'),)

    # Relationships
    recommendation_plan = relationship("RecommendationPlan", back_populates="days")

# Table: recommendation_schedules
class RecommendationSchedule(Base):
    __tablename__ = 'recommendation_schedules'

    schedule_id = Column(BigInteger, primary_key=True, autoincrement=True)
    day_id = Column(BigInteger, ForeignKey('recommended_days.day_id', ondelete="CASCADE"), nullable=False)
    place_id = Column(BigInteger, ForeignKey('recommendation_places.place_id'), nullable=False)
    schedule_order = Column(Integer, nullable=False)

    __table_args__ = (UniqueConstraint('day_id', 'schedule_order', name='_day_order_uc'),)

    # Relationships
    recommended_day = relationship("RecommendedDay", back_populates="schedules")
    place = relationship("RecommendationPlace", back_populates="schedules")

# Table: recommendation_places
class RecommendationPlace(Base):
    __tablename__ = 'recommendation_places'

    place_id = Column(BigInteger, primary_key=True, autoincrement=True)
    place_name = Column(String(100), nullable=False)
    address = Column(String(255), nullable=True)
    latitude = Column(DECIMAL(10, 7), nullable=False)
    longitude = Column(DECIMAL(10, 7), nullable=False)

    # __table_args__ = (UniqueConstraint('latitude', 'longitude', name='_lat_lon_uc'),)

    # Relationships
    schedules = relationship("RecommendationSchedule", back_populates="place")

# Table: preferences
class Preference(Base):
    __tablename__ = 'preferences'

    preference_id = Column(BigInteger, primary_key=True, autoincrement=True)
    recommendation_id = Column(BigInteger, ForeignKey('recommendation_plans.recommendation_trip_id', ondelete="CASCADE"), nullable=False)
    style = Column(Enum('VERY_NATURE', 'MODERATE_NATURE', 'NEUTRAL', 'MODERATE_CITY', 'VERY_CITY'), nullable=False)
    motive = Column(Enum('ESCAPE', 'REST', 'COMPANION_BONDING', 'SELF_REFLECTION', 'SOCIAL_MEDIA', 'EXERCISE', 'NEW_EXPERIENCE', 'CULTURAL_EDUCATION', 'SPECIAL_PURPOSE'), nullable=False)
    means_of_transportation = Column(Enum('CAR', 'PUBLIC_TRANSPORTATION'), nullable=False)
    travel_companion_status = Column(Enum('GROUP_OVER_3', 'WITH_CHILD', 'DUO', 'SOLO', 'FAMILY_DUO', 'EXTENDED_FAMILY'), nullable=False)
    age_group = Column(Enum('UNDER_9', 'TEENS', '20S', '30S', '40S', '50S', '60S', '70_AND_OVER'), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    __table_args__ = (
        CheckConstraint('end_date > start_date', name='check_dates'),
    )

# Table: purposes
class Purpose(Base):
    __tablename__ = 'purposes'

    purposes_id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)

# Table: preference_purposes (many-to-many relation between Preference and Purpose)
class PreferencePurpose(Base):
    __tablename__ = 'preference_purposes'

    preference_id = Column(BigInteger, ForeignKey('preferences.preference_id', ondelete="CASCADE"), primary_key=True, nullable=False)
    purposes_id = Column(BigInteger, ForeignKey('purposes.purposes_id', ondelete="CASCADE"), primary_key=True, nullable=False)

# Relationships
RecommendationPlan.days = relationship("RecommendedDay", order_by=RecommendedDay.day_number, back_populates="recommendation_plan")
RecommendedDay.schedules = relationship("RecommendationSchedule", back_populates="recommended_day")

# Create an engine for the MySQL database
DATABASE_URL = ""
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables in the database (if they don't exist already)
Base.metadata.create_all(bind=engine)
