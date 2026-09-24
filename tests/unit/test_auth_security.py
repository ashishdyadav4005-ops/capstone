"""Unit tests for password hashing and JWT token security."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.auth.jwt import create_access_token, create_refresh_token, decode_token
from src.auth.security import hash_password, verify_password


class TestAuthSecurity:
    """Test suite for password cryptography and JWT token handling."""

    def test_password_hashing_and_verification(self) -> None:
        """Verify that password hashes are non-deterministic (salted) and verifiable."""
        password = "SecurePassword@123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2 # Salted differently
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)
        assert not verify_password("WrongPassword", hash1)
        assert not verify_password("", hash1)

    def test_empty_password_raises_error(self) -> None:
        """Verify that hashing empty password raises ValueError."""
        with pytest.raises(ValueError):
            hash_password("")

    def test_access_token_creation_and_decoding(self) -> None:
        """Verify access token payload, token type, and expiration."""
        payload = {"sub": "analyst", "role": "analyst", "user_id": "USR_001"}
        token = create_access_token(payload, expires_delta=timedelta(minutes=30))

        decoded = decode_token(token)
        assert decoded["sub"] == "analyst"
        assert decoded["role"] == "analyst"
        assert decoded["token_type"] == "access"
        assert "exp" in decoded
        assert "iat" in decoded

    def test_refresh_token_creation_and_decoding(self) -> None:
        """Verify refresh token payload and token type."""
        payload = {"sub": "manager", "role": "manager", "user_id": "USR_002"}
        token = create_refresh_token(payload, expires_delta=timedelta(days=7))

        decoded = decode_token(token)
        assert decoded["sub"] == "manager"
        assert decoded["token_type"] == "refresh"

    def test_expired_token_raises_error(self) -> None:
        """Verify that expired JWT token raises ValueError."""
        # Create token that expired 5 seconds ago
        payload = {"sub": "testuser"}
        token = create_access_token(payload, expires_delta=timedelta(seconds=-5))

        with pytest.raises(ValueError, match="Invalid or expired token"):
            decode_token(token)

    def test_tampered_token_raises_error(self) -> None:
        """Verify that tampered signature raises ValueError."""
        token = create_access_token({"sub": "admin"})
        tampered = token[:-5] + "XXXXX"

        with pytest.raises(ValueError):
            decode_token(tampered)
