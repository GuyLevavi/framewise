"""Alembic entry point: runs migrations against DATABASE_URL."""

from logging.config import fileConfig

from alembic import context

from shared.db import Base, make_engine

if context.config.config_file_name:
    fileConfig(context.config.config_file_name)

with make_engine().connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
