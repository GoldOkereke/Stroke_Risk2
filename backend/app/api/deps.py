# Import Annotated from typing for the modern FastAPI dependency syntax
from typing import Annotated, AsyncGenerator
# Import Depends and HTTPException from FastAPI
# Depends is the injection mechanism; HTTPException is how we return error responses
from fastapi import Depends, HTTPException, status, Request
# Import OAuth2PasswordBearer so FastAPI knows how to extract the Bearer token from headers
from fastapi.security import OAuth2PasswordBearer
# Import AsyncSession and async_sessionmaker for async database sessions
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
# Import select from SQLAlchemy for building queries
from sqlalchemy import select
# Import our security helpers from Day 2
from app.core.security import verify_access_token
# Import the logger from Day 1
from app.core.logging import logger
# Import the User model (assumed to exist from a future Day 3 models file)
# We import it lazily inside functions to avoid circular imports — see below
# from backend.app.models.user import User


# Create the OAuth2 scheme instance
# tokenUrl points to the login endpoint that issues tokens (used by Swagger UI)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# Define the database session dependency as an async generator
async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    # Access the engine stored on app.state during the FastAPI lifespan startup
    # We store it there instead of creating a new engine per request (engines are expensive)
    engine = request.app.state.db_engine
    # Create a session factory bound to this engine
    # expire_on_commit=False keeps object attributes accessible after commit (useful for responses)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    # Open a new session using the factory
    async with session_factory() as session:
        # Wrap in try/except/finally to guarantee cleanup
        try:
            # Yield the session to the route handler — this is where the route runs
            yield session
            # After the route returns successfully, commit the transaction
            await session.commit()
            # Log at DEBUG so we can trace session lifecycle in dev
            logger.debug("Database session committed")
        # Catch any exception raised by the route handler
        except Exception as e:
            # Roll back the transaction so we don't leave partial writes
            await session.rollback()
            # Log the rollback with the exception message
            logger.error(f"Database session rolled back due to: {e}")
            # Re-raise so FastAPI's exception handlers return the correct HTTP status
            raise
        # The `async with` block automatically closes the session on exit
        # This runs whether we committed, rolled back, or the request was cancelled


# Create a reusable type alias for dependency injection
# Annotated[AsyncSession, Depends(get_db)] means "this parameter is an AsyncSession provided by get_db"
SessionDep = Annotated[AsyncSession, Depends(get_db)]


# Define the dependency that extracts and validates the current user from the JWT
async def get_current_user(
    # The token comes from the Authorization header via the OAuth2 scheme
    token: Annotated[str, Depends(oauth2_scheme)],
    # The database session comes from get_db
    db: SessionDep,
):
    # Import User here (not at module top) to avoid circular import:
    # models -> schemas -> deps -> models would loop if imported at top
    from app.db.models import User
    # Define a reusable 401 exception matching FastAPI's expected format
    credentials_exception = HTTPException(
        # 401 means "you are not authenticated"
        status_code=status.HTTP_401_UNAUTHORIZED,
        # "detail" is the message returned in the JSON response body
        detail="Could not validate credentials",
        # WWW-Authenticate header tells the client what scheme to use
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Decode and verify the token is a valid ACCESS token (not refresh)
    payload = verify_access_token(token)
    # If the payload is None, the token was invalid, expired, or wrong type
    if payload is None:
        # Raise the 401 exception
        raise credentials_exception
    # Extract the "sub" claim, which we set to the user ID when creating the token
    user_id: str = payload.get("sub")
    # If "sub" is missing (malformed token), reject
    if user_id is None:
        # Log the anomaly for security monitoring
        logger.warning("Token decoded but missing 'sub' claim")
        # Raise 401
        raise credentials_exception
    # Build a SELECT query to fetch the user by primary key
    # We convert user_id to int because our User.id column is an integer
    result = await db.execute(select(User).where(User.id == int(user_id)))
    # .scalar_one_or_none() returns the User or None (raises if multiple rows — impossible here)
    user = result.scalar_one_or_none()
    # If the user doesn't exist (deleted account, invalid ID in token)
    if user is None:
        # Log the event
        logger.warning(f"Token references non-existent user_id={user_id}")
        # Raise 401
        raise credentials_exception
    # Attach the payload to the user object for convenience (role checks, etc.)
    # Using setattr avoids needing to modify the SQLAlchemy model
    setattr(user, "token_payload", payload)
    # Return the user object so route handlers can use it
    return user


# Create a type alias for routes that need the authenticated user
CurrentUser = Annotated["User", Depends(get_current_user)]


# Define a dependency that additionally checks the user is active
async def get_current_active_user(
    # Depends on get_current_user above
    current_user: CurrentUser,
):
    # Check the is_active column on the User model
    if not current_user.is_active:
        # Return 400 (not 401) because the user IS authenticated, just not allowed
        raise HTTPException(
            # 400 Bad Request is the standard for "authenticated but not authorized"
            status_code=status.HTTP_400_BAD_REQUEST,
            # Clear message for the client
            detail="Inactive user account",
        )
    # If active, return the user
    return current_user


# Create a type alias for routes that need an active authenticated user
ActiveUser = Annotated["User", Depends(get_current_active_user)]


# Define a dependency that requires a specific role (for admin-only routes)
def require_role(required_role: str):
    # This is a "dependency factory" — it returns a dependency function
    async def role_checker(
        # Inject the current active user
        current_user: ActiveUser,
    ):
        # Check the user's role column
        if current_user.role != required_role:
            # Log the authorization failure
            logger.warning(
                f"User {current_user.id} with role '{current_user.role}' "
                f"attempted to access route requiring '{required_role}'"
            )
            # 403 Forbidden = authenticated but not allowed
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}",
            )
        # Role matched — return the user
        return current_user
    # Return the inner function so FastAPI calls it as a dependency
    return role_checker


# Define a dependency to extract the client's real IP (respecting X-Forwarded-For)
def get_client_ip(request: Request) -> str:
    # Check if the request came through a reverse proxy (nginx, Cloudflare, etc.)
    forwarded_for = request.headers.get("X-Forwarded-For")
    # If present, X-Forwarded-For may be a comma-separated list; the first is the original client
    if forwarded_for:
        # Split on comma, take the first entry, strip whitespace
        return forwarded_for.split(",")[0].strip()
    # Fall back to the direct connection IP
    # request.client can be None in some test scenarios
    return request.client.host if request.client else "unknown"


# Create a type alias for the client IP dependency
ClientIP = Annotated[str, Depends(get_client_ip)]