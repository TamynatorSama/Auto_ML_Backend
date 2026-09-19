"""
crypto.py
---------
Sandbox host tokens (and, in Phase 5, model keys) are stored encrypted with
Fernet, keyed by AUTOML_SECRET_KEY. Only the last four characters are kept in
the clear, for display.

A new key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet


def secret_key() -> str:
    key = os.environ.get("AUTOML_SECRET_KEY")
    if not key:
        raise RuntimeError("AUTOML_SECRET_KEY is not set (see worker/crypto.py)")
    return key


def encrypt(key: str, plain: str) -> str:
    return Fernet(key).encrypt(plain.encode()).decode()


def decrypt(key: str, ciphertext: str) -> str:
    return Fernet(key).decrypt(ciphertext.encode()).decode()


def last4(plain: str) -> str:
    return plain[-4:]
