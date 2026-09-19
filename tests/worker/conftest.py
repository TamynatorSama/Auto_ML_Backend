"""Worker tests run against AUTOML_TEST_DATABASE_URL, each in a fresh schema dropped afterwards.

Without that variable they are skipped, so the rest of the suite still runs anywhere.
"""

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from models import DataColumn, DataSchema, DataType
from worker import cli, db

URL = os.environ.get("AUTOML_TEST_DATABASE_URL")


@pytest.fixture
def pool():
    if not URL:
        pytest.skip("AUTOML_TEST_DATABASE_URL is not set")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    pool = db.pool(make_conninfo(URL, options=f"-c search_path={schema}"), size=6)
    try:
        db.setup(pool)
        yield pool
    finally:
        pool.close()
        with psycopg.connect(URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def key():
    return Fernet.generate_key().decode()


@pytest.fixture
def data(tmp_path):
    """A small classification dataset and its schema."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=300)
    frame = pd.DataFrame({
        "id": range(300), "x1": x1, "x2": rng.normal(size=300),
        "color": rng.choice(["red", "green", "blue"], size=300),
        "y": (x1 + rng.normal(scale=0.5, size=300) > 0).astype(int),
    })
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    schema = DataSchema(name="toy", description="a toy classification set", columns=[
        DataColumn(name="id", data_type=DataType.NUMERIC, description="row id", role="identifier"),
        DataColumn(name="x1", data_type=DataType.NUMERIC, description="signal"),
        DataColumn(name="x2", data_type=DataType.NUMERIC, description="noise"),
        DataColumn(name="color", data_type=DataType.CATEGORICAL, description="a colour"),
        DataColumn(name="y", data_type=DataType.CATEGORICAL, description="the label", is_target=True),
    ])
    return str(path), schema


@pytest.fixture
def new_job(pool, key, data):
    """new_job(workspace="default") -> a job id, planning, with its plan task queued."""
    def make(workspace="default"):
        with pool.connection() as conn:
            workspace_id = cli.workspace(conn, workspace)
            if conn.execute("SELECT 1 FROM sandbox_hosts").fetchone() is None:
                cli.add_host(conn, "http://sandbox.test", "a-token-1234", key)
            return cli.create_job(conn, workspace_id, "toy churn", *data)
    return make
