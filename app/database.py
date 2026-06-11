from sqlalchemy import create_engine, Column, String, Float, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

engine = create_engine(
    "sqlite:///./benchmark.db",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class RunRecord(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model = Column(String)
    domain = Column(String)
    accuracy = Column(Float)
    hallucination_rate = Column(Float)
    avg_latency_ms = Column(Float)
    total_tokens = Column(Integer)
    samples_tested = Column(Integer)
    raw_results = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class SampleRecord(Base):
    __tablename__ = "samples"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String)
    model = Column(String)
    domain = Column(String)
    question = Column(Text)
    correct_answer = Column(Text)
    model_answer = Column(Text)
    is_correct = Column(Integer)
    is_hallucination = Column(Integer)
    latency_ms = Column(Float)
    tokens_used = Column(Integer)

Base.metadata.create_all(bind=engine)