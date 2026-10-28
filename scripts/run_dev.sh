#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
cd backend
python -m pip install -q -r requirements.txt
python main.py
