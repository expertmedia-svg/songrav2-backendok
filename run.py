#!/usr/bin/env python3
"""
Lance le backend SONGRA avec encodage UTF-8 forcé
"""
import os
import sys
import subprocess

# Force UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# S'assurer qu'on est dans le bon répertoire
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Les cles API doivent venir de l'environnement local ou d'un .env non versionne.
if not os.getenv('GEMINI_API_KEY') and not os.getenv('OPENAI_API_KEY'):
	print('[WARN] Aucune cle API detectee. Configurez GEMINI_API_KEY et/ou OPENAI_API_KEY avant le lancement.')

# Lance main.py
subprocess.run([sys.executable, 'main.py'])
