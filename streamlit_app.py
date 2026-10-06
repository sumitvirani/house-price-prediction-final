"""
Root Streamlit entrypoint for Streamlit Community Cloud.
Imports and runs the frontend application.
"""

import sys
import os

# Ensure repo root is on path so relative imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

# Re-export app content so  `streamlit run streamlit_app.py` works
exec(open(os.path.join(os.path.dirname(__file__), "frontend", "app.py")).read())
