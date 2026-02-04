-- Initialize databases for local development.
-- This runs automatically in the Postgres container on first startup.
-- Postgres doesn't support CREATE DATABASE IF NOT EXISTS.
-- Ignore errors if databases already exist.
\set ON_ERROR_STOP off
CREATE DATABASE airflow;
CREATE DATABASE mlflow;
\set ON_ERROR_STOP on

