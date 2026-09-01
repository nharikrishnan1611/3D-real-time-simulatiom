from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Use SQLite as a fallback for the hackathon to ensure it runs easily
SQLALCHEMY_DATABASE_URL = "sqlite:///./digital_twin.db"

# To use PostgreSQL, swap to:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Building(Base):
    __tablename__ = "buildings"

    id = Column(String, primary_key=True, index=True)
    type = Column(String)
    estimated_height = Column(Float)
    estimated_elevation = Column(Float)
    polygon = Column(JSON)
    vulnerability_score = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
