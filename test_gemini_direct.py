#!/usr/bin/env python3
"""
Test Gemini API - Valide la connexion et l'analyse photo
Utilise la même API Key configurée dans main.py
"""

import os
import sys
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Charge le fichier .env
import google.generativeai as genai
from PIL import Image
import io

# Couleurs pour le terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_section(title):
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}{title}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}\n")

def test_api_key():
    """Test 1: Vérifier que la clé API est disponible"""
    print_section("TEST 1: Vérification de l'API Key")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key:
        print(f"{RED}❌ ERREUR: GEMINI_API_KEY non définie{RESET}")
        print(f"{YELLOW}Configuration nécessaire:{RESET}")
        print(f"  Windows PowerShell:")
        print(f"    $env:GEMINI_API_KEY='votre_clé_ici'")
        print(f"  Linux/Mac:")
        print(f"    export GEMINI_API_KEY='votre_clé_ici'")
        return False
    
    print(f"{GREEN}✅ GEMINI_API_KEY trouvée{RESET}")
    print(f"   Clé: {gemini_key[:15]}...{gemini_key[-5:]}")
    return True

def test_gemini_connection():
    """Test 2: Vérifier la connexion à Gemini"""
    print_section("TEST 2: Connexion à Gemini API")
    
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_key)
        
        # Lister les modèles disponibles
        print(f"{YELLOW}📋 Modèles disponibles:{RESET}")
        for model in genai.list_models():
            if "vision" in model.name or "gemini" in model.name:
                print(f"   • {model.name}")
        
        print(f"\n{GREEN}✅ Connexion à Gemini établie{RESET}")
        return True
    except Exception as e:
        print(f"{RED}❌ ERREUR de connexion: {str(e)}{RESET}")
        return False

def test_text_generation():
    """Test 3: Tester génération de texte simple"""
    print_section("TEST 3: Génération de Texte")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Réponds en 1 phrase: Que cultive-t-on en agriculture?")
        
        print(f"{YELLOW}Prompt:{RESET} Que cultive-t-on en agriculture?")
        print(f"{YELLOW}Réponse:{RESET} {response.text[:200]}...")
        print(f"\n{GREEN}✅ Génération de texte fonctionne{RESET}")
        return True
    except Exception as e:
        print(f"{RED}❌ ERREUR: {str(e)}{RESET}")
        return False

def create_test_image():
    """Crée une image de test (carré rouge 300x300)"""
    img = Image.new('RGB', (300, 300), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_vision_analysis():
    """Test 4: Tester analyse d'image (vision)"""
    print_section("TEST 4: Vision - Analyse d'Image")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Créer image test
        img_bytes = create_test_image()
        img_bytes.seek(0)
        
        # Test 1: Image simple
        print(f"{YELLOW}Test 4a: Description simple d'image{RESET}")
        response = model.generate_content([
            "Décris cette image en 1 phrase",
            {
                'mime_type': 'image/png',
                'data': base64.standard_b64encode(img_bytes.read()).decode()
            }
        ])
        
        print(f"   Réponse: {response.text[:150]}...")
        img_bytes.seek(0)
        
        print(f"\n{GREEN}✅ Vision fonctionne{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}❌ ERREUR vision: {str(e)}{RESET}")
        return False

def test_agriculture_prompt():
    """Test 5: Tester le prompt agriculture"""
    print_section("TEST 5: Agriculture - Analyse Maladie")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Créer image test
        img_bytes = create_test_image()
        img_bytes.seek(0)
        
        prompt = """[AGRICULTURE ANALYSIS]
Analyse cette image de culture/plant pour détecter maladies ou problèmes.

INSTRUCTIONS:
- Identifie le type de culture si visible
- Détecte les maladies, carences, ou problèmes
- Évalue la gravité et l'urgence
- Propose des traitements

RÉPONDS EN JSON UNIQUEMENT:
{
    "disease_detected": boolean,
    "disease_name": "nom ou null",
    "confidence": 0-1,
    "severity": "none/low/medium/high",
    "symptoms": ["liste des symptômes observés"],
    "treatment": "recommandations de traitement",
    "urgency": "low/medium/high"
}"""
        
        print(f"{YELLOW}Envoi d'image à Gemini pour analyse agriculture...{RESET}")
        
        response = model.generate_content([
            prompt,
            {
                'mime_type': 'image/png',
                'data': base64.standard_b64encode(img_bytes.read()).decode()
            }
        ])
        
        # Essayer de parser JSON
        response_text = response.text
        print(f"{YELLOW}Réponse brute:{RESET}")
        print(f"   {response_text[:300]}...")
        
        try:
            # Chercher JSON dans la réponse
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"\n{YELLOW}JSON parsé avec succès:{RESET}")
                print(f"   Disease detected: {result.get('disease_detected')}")
                print(f"   Confidence: {result.get('confidence')}")
        except:
            pass
        
        print(f"\n{GREEN}✅ Analyse agriculture fonctionne{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}❌ ERREUR analyse: {str(e)}{RESET}")
        return False

def test_gemini_vision_engine():
    """Test 6: Tester la classe GeminiVisionEngine directement"""
    print_section("TEST 6: GeminiVisionEngine Class")
    
    try:
        from gemini_vision import GeminiVisionEngine
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        engine = GeminiVisionEngine(gemini_key)
        
        print(f"{YELLOW}✅ GeminiVisionEngine initialisée{RESET}")
        print(f"   Modèle: {engine.model}")
        
        # Test avec image dummy
        img_bytes = create_test_image()
        img_b64 = base64.standard_b64encode(img_bytes.read()).decode()
        
        result = engine.analyze_images(
            images_data=[img_b64],
            text_description="Image test",
            category="agriculture"
        )
        
        print(f"\n{YELLOW}Résultat d'analyse:{RESET}")
        print(f"   Maladie détectée: {result.get('disease_detected')}")
        print(f"   Confiance: {result.get('confidence')}")
        print(f"   Modèle utilisé: {result.get('model')}")
        
        print(f"\n{GREEN}✅ GeminiVisionEngine fonctionne correctement{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}❌ ERREUR GeminiVisionEngine: {str(e)}{RESET}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Orchestre tous les tests"""
    print(f"\n{BOLD}{BLUE}🚀 TEST GEMINI API - RESOLVE HUB{RESET}\n")
    
    results = {
        "API Key": test_api_key(),
        "Connexion": test_gemini_connection(),
        "Texte": test_text_generation(),
        "Vision": test_vision_analysis(),
        "Agric": test_agriculture_prompt(),
        "Engine": test_gemini_vision_engine(),
    }
    
    # Résumé
    print_section("📊 RÉSUMÉ DES TESTS")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
        print(f"  {status} {name}")
    
    print(f"\n{BLUE}Score: {passed}/{total}{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{BOLD}🎉 Tous les tests Gemini passés!{RESET}")
        print(f"{GREEN}Tu peux lancer le backend avec: python main.py{RESET}\n")
        return 0
    else:
        print(f"\n{YELLOW}{BOLD}⚠️  Certains tests ont échoué{RESET}")
        print(f"{RED}Vérifie ta clé API et ta connexion réseau{RESET}\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
