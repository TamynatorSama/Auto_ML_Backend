"""
sources.py
----------
The worker's source tasks (docs/PHASE4.md §4). None of them calls an LLM.

    ingest   streams the CSV to check it parses and to count its rows and
             columns; light, so it doesn't take a heavy slot
    profile  profile_dataset over the whole file -> sources.profile, and a
             draft schema filled in from it; reads the file into pandas, so it
             shares the heavy cap with plan

Both run here rather than in a sandbox: this is the pipeline's own trusted
code, and a sandbox would only add a round trip of the data.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from api import files   # only for where uploads live: the one rule, shared (docs/PHASE4.md §10)
from db import jsonb
from models import DataColumn, DataSchema, DataType
from utils.reusable.summary import profile_dataset
from worker import queue

# the profile's inferred role -> what the draft schema says about the column
ROLE_TYPES = {
    "numeric": DataType.NUMERIC,
    "discrete_numeric": DataType.NUMERIC,
    "binary": DataType.CATEGORICAL,
    "categorical": DataType.CATEGORICAL,
    "text": DataType.TEXT,
    "datetime": DataType.TEXT,       # the pipeline has no date type; the description carries it
    "identifier": DataType.TEXT,
    "empty": DataType.TEXT,
    "constant": DataType.TEXT,
}
SEMANTIC_TYPES = {"latitude": DataType.LATITUDE, "longitude": DataType.LONG}
# a column with nothing in it, or the same value throughout, teaches a model nothing
IGNORED_ROLES = {"empty", "constant"}


def load(conn, source_id: int) -> dict:
    return conn.execute("SELECT * FROM sources WHERE id = %s", (source_id,)).fetchone()


def ingest(pool, saver, task: dict) -> None:
    """Parse the file once, streaming: a header with unique names, and the row count."""
    source_id = task["source_id"]
    with pool.connection() as conn:
        source = load(conn, source_id)
    rows, columns = read_shape(Path(source["path"]))

    with pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE sources SET status = 'profiling', rows = %s, columns = %s WHERE id = %s",
                (rows, len(columns), source_id),
            )
            queue.enqueue(conn, "profile", source_id=source_id)


def read_shape(path: Path) -> tuple[int, list]:
    """(data rows, column names). Raises ValueError on anything that isn't a usable CSV."""
    # a 100 MB file is parsed a row at a time, so memory stays flat
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except (StopIteration, csv.Error) as error:
            raise ValueError("the file is empty or is not a CSV") from error
        check_header(header)
        try:
            rows = sum(1 for row in reader if row)
        except csv.Error as error:
            raise ValueError(f"the file stopped parsing as a CSV: {error}") from error
    if not rows:
        raise ValueError("the file has a header but no rows")
    return rows, header


def check_header(header: list) -> None:
    names = [name.strip() for name in header]
    if not names or not any(names):
        raise ValueError("the first row must name the columns")
    if "" in names:
        raise ValueError(f"column {names.index('') + 1} has no name")
    seen = {name for name in names if names.count(name) > 1}
    if seen:
        raise ValueError(f"these column names appear more than once: {', '.join(sorted(seen))}")


def profile(pool, saver, task: dict) -> None:
    """The whole file into pandas, profiled, and a draft schema written from the profile."""
    import pandas as pd

    source_id = task["source_id"]
    with pool.connection() as conn:
        source = load(conn, source_id)
    frame = pd.read_csv(source["path"])
    report = profile_dataset(frame)
    schema = draft_schema(source["name"], report)

    with pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE sources SET status = 'ready', profile = %s, rows = %s, columns = %s, error = NULL "
                "WHERE id = %s",
                (jsonb(report), len(frame), len(frame.columns), source_id),
            )
            write_draft(conn, source_id, schema)


def draft_schema(name: str, report: dict) -> DataSchema:
    """The profile's own roles and semantic types, as a schema for the person to correct.

    Descriptions start empty: they reach every prompt the pipeline writes, and
    only the person knows what a column means (G12).
    """
    columns = [
        DataColumn(
            name=column_name,
            data_type=column_type(column),
            description="",
            role="identifier" if column["role"] == "identifier"
                 else "ignore" if column["role"] in IGNORED_ROLES else "feature",
        )
        for column_name, column in report["columns"].items()
    ]
    return DataSchema(name=name, columns=columns, description="")


def column_type(column: dict) -> DataType:
    semantic = SEMANTIC_TYPES.get(column.get("semantic_type"))
    return semantic or ROLE_TYPES.get(column["role"], DataType.TEXT)


def write_draft(conn, source_id: int, schema: DataSchema) -> int:
    """The draft, written or refreshed; a locked version is never touched.

    What a person — or the shipped sample — already said about a column outlives
    the profile: profiling again fills in the facts and keeps the meaning (§8.4).
    """
    old = conn.execute(
        "SELECT id, columns FROM schemas WHERE source_id = %s AND status = 'draft'", (source_id,)
    ).fetchone()
    if old:
        schema = keep_edits(schema, old["columns"])
        conn.execute("UPDATE schemas SET columns = %s, updated_at = now() WHERE id = %s",
                     (jsonb(schema.model_dump(mode="json")), old["id"]))
        return old["id"]
    version = conn.execute(
        "SELECT coalesce(max(version), 0) + 1 AS next FROM schemas WHERE source_id = %s", (source_id,)
    ).fetchone()["next"]
    return conn.execute(
        "INSERT INTO schemas (source_id, version, status, columns) VALUES (%s, %s, 'draft', %s) RETURNING id",
        (source_id, version, jsonb(schema.model_dump(mode="json"))),
    ).fetchone()["id"]


def keep_edits(schema: DataSchema, existing: dict) -> DataSchema:
    """Carry what was already said about each column still in the file onto the fresh draft."""
    said = {column["name"]: column for column in existing.get("columns", [])}
    for column in schema.columns:
        was = said.get(column.name)
        if not was:
            continue
        column.description = was.get("description") or column.description
        column.available_at_prediction = was.get("available_at_prediction")
        column.data_type = DataType(was["data_type"])
        column.role = was["role"]
        column.is_target = was["role"] == "target"
    schema.description = existing.get("description") or schema.description
    return schema


def analyze(pool, saver, task: dict) -> None:
    """The leakage screen, once a target is chosen (§5). Still no LLM.

    Profiles a second time, now that there is a target, so the target's own
    statistics and the per-column associations exist (§8.3); then asks whether a
    simple formula of one or two columns reproduces the target on held-out rows.
    """
    import pandas as pd

    from utils.reusable.leakage import relation_screen

    source_id = task["source_id"]
    with pool.connection() as conn:
        source = load(conn, source_id)
        draft = conn.execute(
            "SELECT id, columns FROM schemas WHERE source_id = %s AND status = 'draft'", (source_id,)
        ).fetchone()
    if draft is None:
        raise ValueError("this dataset has no draft schema to analyse")
    schema = DataSchema.model_validate(draft["columns"])
    target = next((column.name for column in schema.columns if column.role == "target"), None)
    if target is None:
        raise ValueError("choose a target column first")

    frame = pd.read_csv(source["path"])
    report = profile_dataset(frame, target=target)
    kinds = [{"name": name, "kind": column["role"]} for name, column in report["columns"].items()]
    task_type = (report.get("target") or {}).get("task_type") or "regression"
    hits = relation_screen(frame, target, task_type, kinds)

    analysis = {"target": target, "findings": findings(report, hits), "acknowledged": []}
    with pool.connection() as conn:
        with conn.transaction():
            conn.execute("UPDATE sources SET profile = %s WHERE id = %s", (jsonb(report), source_id))
            conn.execute("UPDATE schemas SET analysis = %s, updated_at = now() WHERE id = %s",
                         (jsonb(analysis), draft["id"]))


def findings(report: dict, hits: list) -> list:
    """What the Schema screen warns about: the screen's hits, then the profile's own flags.

    Both come from the data, so a dataset we have never seen warns about itself.
    """
    out = [
        {
            "id": "leak:" + ",".join(hit["columns"]),
            "severity": "critical",
            "columns": hit["columns"],
            # the screen prints the columns itself, so the sentence starts after them
            "issue": f"reproduces the target on held-out rows ({hit['measure']} {hit['score']:.3f})"
                     + (f": {hit['formula']}" if hit.get("formula") else ""),
            "action": "If these are known at prediction time, say so and keep them; otherwise mark them ignore.",
        }
        for hit in hits
    ]
    for flag in report.get("flags", []):
        columns = [flag["column"]] if flag.get("column") else []
        out.append({
            "id": f"flag:{flag.get('column') or ''}:{flag['issue'][:40]}",
            "severity": flag["severity"],
            "columns": columns,
            "issue": flag["issue"],
            "action": flag.get("action", ""),
        })
    return out


def readable(error: str) -> str:
    """What the person is shown. Where the file sits on our disk is not their business.

    A pandas or OS error names the path it was reading; the person uploaded one
    file and knows which, so the path says nothing to them and something to
    anyone else. The worker's own log keeps the full text.
    """
    upload = re.escape(str(files.UPLOADS_ROOT))
    return re.sub(rf"['\"]?{upload}[^'\"\s]*['\"]?", "the file", error)[:2000]


def fail(pool, source_id: int, error: str, kind: str | None = None) -> None:
    """A failed ingest or profile leaves no usable source; a failed analyze leaves the file fine."""
    error = readable(error)
    with pool.connection() as conn:
        if kind == "analyze":
            conn.execute("UPDATE sources SET error = %s WHERE id = %s", (error, source_id))
        else:
            conn.execute("UPDATE sources SET status = 'failed', error = %s WHERE id = %s",
                         (error, source_id))


HANDLERS = {"ingest": ingest, "profile": profile, "analyze": analyze}
