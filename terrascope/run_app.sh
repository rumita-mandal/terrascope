#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m uvicorn terrascope.server.app:app --host 0.0.0.0 --port ${PORT:-8000}
