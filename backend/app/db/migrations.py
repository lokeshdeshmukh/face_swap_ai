from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def ensure_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("jobs"):
        return

    columns = {column["name"] for column in inspector.get_columns("jobs")}
    statements: list[str] = []

    if "input_config_json" not in columns:
        statements.append("ALTER TABLE jobs ADD COLUMN input_config_json TEXT NOT NULL DEFAULT '{}'")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
