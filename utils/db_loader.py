# db_operation.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import HTTPException
from datetime import datetime

from utils.config_loader import load_config
from logger import GLOBAL_LOGGER as log
from workflow.responsemodel import responsemodel  # adjust as needed


class SqliteDBLoader:
    """
    SQLite DB loader that encapsulates engine, session, and metadata setup.
    """
    def __init__(self):
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        self.database_url = self.config["sqlite"]["SQLITE_Userdb_URL"]
        log.info(f"Connecting to SQLite DB at: {self.database_url}")

        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False}
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        self.Base = declarative_base()

    def get_db(self):
        """
        FastAPI dependency to provide DB session.
        """
        db: Session = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self):
        """
        Initialize all tables.
        """
        self.Base.metadata.create_all(bind=self.engine)
        log.info("Database tables created successfully.")


# Instantiate loader
db_loader = SqliteDBLoader()
get_db = db_loader.get_db
Base = db_loader.Base
engine = db_loader.engine


# ---------- SQLAlchemy Model ----------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    department = Column(String(255), nullable=False)
    username = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------- Repository Layer ----------
class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def register_user(self, user_data: responsemodel.UserRegistration) -> responsemodel.SuccessResponse:
        try:
            db_user = User(
                email=user_data.email,
                department=user_data.department,
                username=user_data.username,
                is_active=True
            )
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)

            return responsemodel.SuccessResponse(
                success=True,
                message="User registered successfully",
                user_id=db_user.id
            )
        except Exception:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Internal server error during registration"
            )
        
    def fetch_all_user(self):
        try:
            users = self.db.query(User).filter(User.is_active == True).all()
            return users
        except Exception as e:
            #logger.error(f"Error fetching users: {str(e)}")
            raise HTTPException(status_code=500, detail="Error fetching users")            

    def fetch_user(self,email):
        try:
            print(email)
            user = self.db.query(User).filter(User.email == email, User.is_active == True).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user
        except HTTPException:
            raise
        except Exception as e:
            #logger.error(f"Error fetching user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error fetching user")

# Optional: Table creation script
if __name__ == "__main__":
    db_loader.init_db()
