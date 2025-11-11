#!/usr/bin/env bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<'EOSQL'
ALTER SYSTEM SET shared_preload_libraries = 'age';
EOSQL

