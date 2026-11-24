# Import datetime and timedelta for calculating token expiry times
from datetime import datetime, timedelta, timezone
# Import Optional and Union for type hints on function signatures
from typing import Optional, Union, Any
# Import jwt from python-jose for encoding/decoding JSON Web Tokens
from jose import jwt, JWTError
# Import CryptContext from passlib to handle bcrypt password hashing
from passlib.context import CryptContext
# Import settings singleton created on Day 1 so we can read SECRET_KEY and ALGORITHM
from app.core.config import settings
# Import the logger we configured on Day 1 so every security event is logged
from app.core.logging import logger


# --- BCRYPT SHIM (fix for passlib + bcrypt 4.x incompatibility) ---
# passlib expects bcrypt.__about__.__version__ but bcrypt 4.x removed it
# We wrap the import in try/except so it doesn't crash if bcrypt is v3
try:
    # Attempt to import bcrypt to check its version attribute
    import bcrypt
    # If the __about__ attribute is missing (bcrypt 4.x), create a fake one
    if not hasattr(bcrypt, "__about__"):
        # Define a tiny namespace object to hold the version string
        class _About:
            # Hardcode the version so passlib's version check passes
            __version__ = bcrypt.__version__
        # Attach the fake __about__ namespace to the bcrypt module
        bcrypt.__about__ = _About()
# Catch any import error silently — if bcrypt isn't installed yet, we'll fail later anyway
except ImportError:
    # Do nothing; the CryptContext below will raise a clearer error
    pass


# Create a CryptContext instance that tells passlib which hashing scheme to use
pwd_context = CryptContext(
    # Use bcrypt as the only (and default) hashing scheme
    schemes=["bcrypt"],
    # "auto" means: if we later change schemes, old hashes are marked deprecated and rehashed on login
    deprecated="auto",
    # Cost factor 12 means 2^12 = 4096 iterations; takes ~250-300ms on modern hardware
    bcrypt__rounds=12,
)


# Define the public function to hash a plain password
def hash_password(password: str) -> str:
    # Log at DEBUG level that hashing is starting (never log the password itself)
    logger.debug("Hashing password")
    # Check if password is empty or None and raise an error early
    if not password:
        # Raise ValueError with a clear message so the caller knows what went wrong
        raise ValueError("Password cannot be empty")
    # bcrypt only uses the first 72 bytes; anything longer is silently ignored
    # This is a security issue because "a"*72 + "extra" and "a"*72 verify the same
    # Encode to bytes to measure length correctly (multi-byte UTF-8 chars count as 1 byte per char)
    if len(password.encode("utf-8")) > 72:
        # Log a warning so developers know truncation is happening
        logger.warning("Password exceeds 72 bytes; truncating for bcrypt")
        # Truncate to exactly 72 bytes, then decode back to str for passlib
        # We slice on bytes then decode to avoid cutting a multi-byte char in half
        password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    # Call passlib's hash method which generates a random salt and returns the full hash string
    # The returned string looks like: $2b$12$<22-char-salt><31-char-hash>
    hashed = pwd_context.hash(password)
    # Log success at DEBUG (again, never log the actual hash in production logs)
    logger.debug("Password hashed successfully")
    # Return the hash string to the caller
    return hashed


# Define the public function to verify a plain password against a stored hash
def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Wrap in try/except because passlib can raise on malformed hashes
    try:
        # passlib extracts the salt from hashed_password, hashes plain_password with it, and compares
        # Returns True if they match, False otherwise
        is_valid = pwd_context.verify(plain_password, hashed_password)
        # Log the outcome at DEBUG level (INFO on failure might spam logs during brute-force attempts)
        logger.debug(f"Password verification result: {is_valid}")
        # Return the boolean result to the caller
        return is_valid
    # Catch any exception passlib might throw (bad hash format, unsupported scheme, etc.)
    except Exception as e:
        # Log the error with the exception message so we can debug malformed hashes
        logger.error(f"Password verification failed with error: {e}")
        # Return False instead of raising, so the login endpoint just says "invalid credentials"
        return False


# Define the function to create a short-lived access token
def create_access_token(
    # subject is typically the user ID (string or int)
    subject: Union[str, int],
    # expires_delta lets callers override the default 15-minute expiry if needed
    expires_delta: Optional[timedelta] = None,
    # extra_claims lets callers add custom fields like "role" or "email" to the payload
    extra_claims: Optional[dict] = None,
) -> str:
    # Get the current time in UTC (timezone-aware, not the deprecated utcnow())
    now = datetime.now(timezone.utc)
    # If the caller provided a custom expiry delta, use it; otherwise default to 15 minutes
    expire = now + (expires_delta if expires_delta else timedelta(minutes=15))
    # Build the JWT payload dictionary with standard claims
    payload = {
        # "sub" (subject) is the standard claim for "who this token is about"
        "sub": str(subject),
        # "exp" (expiry) tells the decoder when the token becomes invalid
        "exp": expire,
        # "iat" (issued at) helps with audit trails and token rotation logic
        "iat": now,
        # "type" is a custom claim preventing refresh tokens from being used as access tokens
        "type": "access",
    }
    # If extra claims were provided, merge them into the payload
    if extra_claims:
        # .update() mutates the dict; later keys overwrite earlier ones
        # We do this AFTER the standard claims so a caller can't accidentally overwrite "exp"
        # Actually: it CAN overwrite exp, but that's the caller's responsibility
        payload.update(extra_claims)
    # Encode the payload into a signed JWT string using the secret key and algorithm
    # This does NOT encrypt the payload — anyone can decode it — but they cannot forge it
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    # Log that a token was issued (log the subject, never the token itself)
    logger.info(f"Access token issued for subject={subject}")
    # Return the JWT string to the caller
    return token


# Define the function to create a long-lived refresh token
def create_refresh_token(
    # Same subject parameter as access token
    subject: Union[str, int],
    # Default refresh expiry is 7 days unless overridden
    expires_delta: Optional[timedelta] = None,
    # Extra claims optional here too
    extra_claims: Optional[dict] = None,
) -> str:
    # Get current UTC time
    now = datetime.now(timezone.utc)
    # Default to 7 days if no custom delta provided
    expire = now + (expires_delta if expires_delta else timedelta(days=7))
    # Build refresh-specific payload
    payload = {
        # Subject is the same user ID
        "sub": str(subject),
        # Expiry is much further in the future
        "exp": expire,
        # Issued at timestamp
        "iat": now,
        # CRITICAL: "type" is "refresh" not "access" — this is the multi-stage part
        "type": "refresh",
    }
    # Merge extra claims if provided
    if extra_claims:
        payload.update(extra_claims)
    # Encode with same secret and algorithm
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    # Log issuance with subject only
    logger.info(f"Refresh token issued for subject={subject}")
    # Return the JWT string
    return token


# Define the function to decode and validate a token
def decode_token(token: str) -> Optional[dict]:
    # Wrap in try/except because jwt.decode raises on invalid/expired tokens
    try:
        # Decode the token, which simultaneously:
        # 1. Verifies the signature (proves we issued it)
        # 2. Checks exp (expired tokens raise ExpiredSignatureError)
        # 3. Validates the algorithm matches
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        # Log successful decode at DEBUG level
        logger.debug(f"Token decoded successfully for sub={payload.get('sub')}")
        # Return the payload dict to the caller
        return payload
    # Catch JWTError, which is the parent class of all jose decode errors
    except JWTError as e:
        # Log the failure at WARNING level (invalid tokens are common in production)
        logger.warning(f"Token decode failed: {e}")
        # Return None so callers can handle "no valid token" without try/except
        return None


# Define a function to specifically validate access tokens (not refresh tokens)
def verify_access_token(token: str) -> Optional[dict]:
    # Decode the token first
    payload = decode_token(token)
    # If decode failed, short-circuit and return None
    if payload is None:
        # Nothing more to do
        return None
    # Check the "type" claim to prevent refresh-token-as-access-token attacks
    if payload.get("type") != "access":
        # Log the security event — this could be an attack attempt
        logger.warning(f"Token type mismatch: expected 'access', got '{payload.get('type')}'")
        # Reject the token
        return None
    # Token is valid and is an access token — return the payload
    return payload


# Define a function to validate refresh tokens specifically
def verify_refresh_token(token: str) -> Optional[dict]:
    # Decode the token
    payload = decode_token(token)
    # Bail out if decode failed
    if payload is None:
        return None
    # Enforce the refresh type
    if payload.get("type") != "refresh":
        # Log the mismatch
        logger.warning(f"Token type mismatch: expected 'refresh', got '{payload.get('type')}'")
        # Reject
        return None
    # Return the valid refresh payload
    return payload