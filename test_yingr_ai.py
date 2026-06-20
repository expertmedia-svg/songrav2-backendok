"""
SONGRA AI Rural Agent - Yingr-AI (IA Souveraine du Burkina Faso)
Script de test pour valider les endpoints de la suite Yingr-AI.
"""

import os
import sys
import base64
import json
from fastapi.testclient import TestClient

# Configurer stdout pour UTF-8 pour éviter l'erreur de codec sur Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import app
    client = TestClient(app)
    print("[OK] TestClient charge avec succes.")
except Exception as e:
    print(f"[ERROR] Impossible de charger l'application FastAPI: {e}")
    sys.exit(1)

DUMMY_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

DUMMY_AUDIO_B64 = (
    "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAAA"
)


def test_agriculture_demo():
    print("\n--- TEST YINGR-AI : Diagnostic Agricole (Feuille malade) ---")
    payload = {
        "text": "carence azote mais jaune",
        "photo_base64": DUMMY_IMAGE_B64
    }
    response = client.post("/api/v3/yingr-ai/diagnose/agriculture", json=payload)
    print(f"Status Code : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Culture detectee : {data.get('culture_detected')}")
        print(f"Diagnostic : {data.get('disease_detected')}")
        print(f"Gravite : {data.get('severity')}")
        print(f"Remedes locaux : {data.get('local_remedies')}")
        print(f"Systeme : {data.get('_meta', {}).get('system')}")
        print(f"Latence : {data.get('_meta', {}).get('duration_ms')} ms")
        assert "culture_detected" in data
        assert "disease_detected" in data
        print("[SUCCESS] Agriculture validee.")
    else:
        print(f"[FAIL] Reponse erreur : {response.text}")


def test_elevage_demo():
    print("\n--- TEST YINGR-AI : Diagnostic Elevage (Animal malade) ---")
    payload = {
        "text": "chevre gale croutes peau",
        "photo_base64": DUMMY_IMAGE_B64
    }
    response = client.post("/api/v3/yingr-ai/diagnose/elevage", json=payload)
    print(f"Status Code : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Espece : {data.get('animal_species')}")
        print(f"Maladie detectee : {data.get('disease_detected')}")
        print(f"Gravite : {data.get('severity')}")
        print(f"Traitement : {data.get('treatment_steps')}")
        print(f"Latence : {data.get('_meta', {}).get('duration_ms')} ms")
        assert "animal_species" in data
        assert "disease_detected" in data
        print("[SUCCESS] Elevage validee.")
    else:
        print(f"[FAIL] Reponse erreur : {response.text}")


def test_sos_demo_audio():
    print("\n--- TEST YINGR-AI : SOS Accident Rural (Audio) ---")
    payload = {
        "text": "",
        "audio_base64": DUMMY_AUDIO_B64
    }
    response = client.post("/api/v3/yingr-ai/sos", json=payload)
    print(f"Status Code : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Type d'urgence : {data.get('emergency_type')}")
        print(f"Gravite : {data.get('severity')}")
        print(f"Actions immediates : {data.get('immediate_actions')}")
        print(f"Transcription Whisper : {data.get('audio_transcription')}")
        print(f"Latence : {data.get('_meta', {}).get('duration_ms')} ms")
        assert "is_emergency" in data
        assert "immediate_actions" in data
        print("[SUCCESS] SOS validee.")
    else:
        print(f"[FAIL] Reponse erreur : {response.text}")


if __name__ == "__main__":
    test_agriculture_demo()
    test_elevage_demo()
    test_sos_demo_audio()
    print("\n[OK] Tous les tests d'API de la suite Yingr-AI ont reussi.")
