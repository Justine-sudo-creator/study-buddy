#!/bin/bash
# Start Study Buddy — run from ~/Downloads/study_workspace
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
