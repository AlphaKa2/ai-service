from sqlalchemy import Column, String, BigInteger, Enum, DateTime, Integer, Date, ForeignKey, DECIMAL, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Table: recommendation_plans
class RecommendationPlan(Base):
    __tablename__ = 'recommendation_plans'
    recommendation_trip_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(String(200), nullable=False)
    recommendation_type = Column(Enum('AI-GENERATED', 'YOUTUBER_FOLLOW'), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    __table_args__ = (
        CheckConstraint('end_date >= start_date', name='check_dates'),
    )

    # Relationships
    days = relationship(
        "RecommendationDay",
        back_populates="recommendation_plan",
        cascade="all, delete-orphan",
    )
    youtube_videos = relationship(
        "TravelPlanYoutubeVideo",
        back_populates="travel_plan",
        cascade="all, delete-orphan",
    )
    preferences = relationship(
        "Preference",
        back_populates="recommendation_plan",
        cascade="all, delete-orphan",
    )

# Table: recommendation_days
class RecommendationDay(Base):
    __tablename__ = 'recommendation_days'
    day_id = Column(BigInteger, primary_key=True, autoincrement=True)
    recommendation_trip_id = Column(
        BigInteger,
        ForeignKey('recommendation_plans.recommendation_trip_id', ondelete="CASCADE"),
        nullable=False
    )
    day_number = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    __table_args__ = (
        UniqueConstraint('recommendation_trip_id', 'day_number', name='_trip_day_uc'),
    )

    # Relationships
    recommendation_plan = relationship("RecommendationPlan", back_populates="days")
    schedules = relationship(
        "RecommendationSchedule",
        back_populates="recommendation_day",
        cascade="all, delete-orphan",
    )

# Table: recommendation_schedules
class RecommendationSchedule(Base):
    __tablename__ = 'recommendation_schedules'
    schedule_id = Column(BigInteger, primary_key=True, autoincrement=True)
    day_id = Column(
        BigInteger,
        ForeignKey('recommendation_days.day_id', ondelete="CASCADE"),
        nullable=False
    )
    place_id = Column(BigInteger, ForeignKey('recommendation_places.place_id', ondelete="CASCADE"), nullable=False)
    schedule_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    recommendation_day = relationship("RecommendationDay", back_populates="schedules")
    place = relationship("RecommendationPlace", back_populates="schedules")

# Table: recommendation_places
class RecommendationPlace(Base):
    __tablename__ = 'recommendation_places'
    place_id = Column(BigInteger, primary_key=True, autoincrement=True)
    place_name = Column(String(100), nullable=False)
    content = Column(String(500), nullable=True)
    address = Column(String(255), nullable=True)
    latitude = Column(DECIMAL(10, 7), nullable=True)
    longitude = Column(DECIMAL(10, 7), nullable=True)

    # Relationships
    schedules = relationship("RecommendationSchedule", back_populates="place")

# Table: preferences
class Preference(Base):
    __tablename__ = 'preferences'
    preference_id = Column(BigInteger, primary_key=True, autoincrement=True)
    recommendation_id = Column(
        BigInteger,
        ForeignKey('recommendation_plans.recommendation_trip_id', ondelete="CASCADE"),
        nullable=False
    )
    style = Column(Enum('VERY_NATURE', 'MODERATE_NATURE', 'NEUTRAL', 'MODERATE_CITY', 'VERY_CITY'), nullable=False)
    motive = Column(Enum('ESCAPE', 'REST', 'COMPANION_BONDING', 'SELF_REFLECTION', 'SOCIAL_MEDIA', 'EXERCISE', 'NEW_EXPERIENCE', 'CULTURAL_EDUCATION', 'SPECIAL_PURPOSE'), nullable=False)
    means_of_transportation = Column(Enum('CAR', 'PUBLIC_TRANSPORTATION'), nullable=False)
    travel_companion_status = Column(Enum('GROUP_OVER_3', 'WITH_CHILD', 'DUO', 'SOLO', 'FAMILY_DUO', 'EXTENDED_FAMILY'), nullable=False)
    age_group = Column(Enum('UNDER_9', 'TEENS', '20S', '30S', '40S', '50S', '60S', '70_AND_OVER'), nullable=False)
    travel_status_days = Column(Integer, nullable=False)
    road_addr = Column(String(50), nullable=False)
    gender = Column(Enum('MALE', 'FEMALE'), nullable=False)

    # Relationships
    recommendation_plan = relationship("RecommendationPlan", back_populates="preferences")
    purposes = relationship(
        "PreferencePurpose",
        back_populates="preference",
        cascade="all, delete-orphan",
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
    preference = relationship("Preference", back_populates="purposes")

# Table: youtubers
class Youtuber(Base):
    __tablename__ = 'youtubers'
    youtuber_id = Column(String(500), primary_key=True)
    name = Column(String(500), nullable=False)
    url = Column(String(500), nullable=False)

    # Relationships
    videos = relationship("YoutubeVideo", back_populates="youtuber")

# Table: youtube_videos
class YoutubeVideo(Base):
    __tablename__ = 'youtube_videos'
    video_id = Column(String(500), primary_key=True)
    youtuber_id = Column(String(500), ForeignKey('youtubers.youtuber_id', ondelete="CASCADE"), nullable=False)
    title = Column(String(100), nullable=False)
    url = Column(String(500), nullable=False)

    # Relationships
    youtuber = relationship("Youtuber", back_populates="videos")
    travel_plans = relationship(
        "TravelPlanYoutubeVideo",
        back_populates="youtube_video",
        cascade="all, delete-orphan",
    )

# Table: travel_plan_youtube_videos
class TravelPlanYoutubeVideo(Base):
    __tablename__ = 'travel_plan_youtube_videos'
    travel_id = Column(
        BigInteger,
        ForeignKey('recommendation_plans.recommendation_trip_id', ondelete="CASCADE"),
        primary_key=True,
        nullable=False
    )
    video_id = Column(
        String(500),
        ForeignKey('youtube_videos.video_id', ondelete="CASCADE"),
        primary_key=True,
        nullable=False
    )

    # Relationships
    travel_plan = relationship("RecommendationPlan", back_populates="youtube_videos")
    youtube_video = relationship("YoutubeVideo", back_populates="travel_plans")

# Database setup
DATABASE_URL = "url"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
