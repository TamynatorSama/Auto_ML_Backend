import pytest
from cryptography.fernet import Fernet, InvalidToken

from worker import crypto


def test_a_token_round_trips_and_only_its_last_four_show(key):
    ciphertext = crypto.encrypt(key, "s3cret-token-abcd")
    assert "s3cret" not in ciphertext
    assert crypto.decrypt(key, ciphertext) == "s3cret-token-abcd"
    assert crypto.last4("s3cret-token-abcd") == "abcd"


def test_another_key_cannot_read_it(key):
    ciphertext = crypto.encrypt(key, "s3cret-token-abcd")
    with pytest.raises(InvalidToken):
        crypto.decrypt(Fernet.generate_key().decode(), ciphertext)
