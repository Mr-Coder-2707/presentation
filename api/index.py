import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable for Vercel
os.environ['VERCEL'] = '1'

# Import the Flask app
from app import app

# Export for Vercel
application = app
