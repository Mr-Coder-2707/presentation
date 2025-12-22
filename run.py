#!/usr/bin/env python
"""Simple script to run the Flask app"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Information Retrieval System - Web Server")
    print("="*60)
    print("\nStarting Flask server on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000, host='127.0.0.1')

