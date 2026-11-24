# core/config.py — What It Should Do

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (backend/app/core/config.py -> backend/app/core -> app -> backend -> root),
# so .env is found no matter which directory the process starts in.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Define a Settings class inheriting from BaseSettings
class Settings(BaseSettings):
    # Declare typed fields (app_name: str, log_level: str, etc.)
    app_name: str = "stroke-backend"
    environment: str = "development"
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    # Secret key for JWT signing — generate with: openssl rand -hex 32
    SECRET_KEY: str
    # Algorithm for JWT signing — HS256 is symmetric (single secret), RS256 is asymmetric
    ALGORITHM: str = "HS256"
    # Access token lifetime in minutes
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    # Refresh token lifetime in days
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Read from the project-root .env, independent of the working directory
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8"
    )

    # Validation to add: Make log_level only accept "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL". Use Pydantic's field_validator decorator.
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        log_level = value.upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in allowed_levels:
            raise ValueError(
                f"log_level must be one of: {', '.join(sorted(allowed_levels))}"
            )
        return log_level


# Export a singleton instance (settings = Settings())
# Key decision: Why a singleton? So every file imports the same settings object, and validation runs once at startup.
settings = Settings()

# Search terms to look up:
# pydantic-settings BaseSettings model_config env_file
# pydantic v2 field_validator example
