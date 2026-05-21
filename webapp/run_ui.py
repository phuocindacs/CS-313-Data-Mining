"""Streamlit launcher — sets asyncio WindowsSelectorEventLoopPolicy before
Streamlit initializes its event loop (required for Python 3.12+ on Windows).
Run with: python run_ui.py
"""

import asyncio
import sys
import os

# Must happen BEFORE any streamlit import
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure webapp/ is in path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from streamlit.web import cli as stcli

sys.argv = [
    "streamlit", "run", "ui/app.py",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.fileWatcherType", "none",
]
sys.exit(stcli.main())
