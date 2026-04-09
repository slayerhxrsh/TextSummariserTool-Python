import sys
from pathlib import Path

# Add project root to sys.path so Vercel can find app.py and text_summarizer
# Path(__file__).parent is 'api', .parent.parent is root
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from app import app
