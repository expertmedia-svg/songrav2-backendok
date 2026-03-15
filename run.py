#!/usr/bin/env python3
"""
Lance le backend SONGRA avec encodage UTF-8 forcé
"""
import os
import sys
import subprocess

# Force UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['GEMINI_API_KEY'] = 'AIzaSyAbuiMBXHE6WWtwzYiwEqqKdi4KaXwrbhE'

# Lance main.py
subprocess.run([sys.executable, 'main.py'])
