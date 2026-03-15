#!/usr/bin/env python3
"""
Test Gemini pour les 4 MODULES du Scanner
1. AGRICULTURE
2. ELEVAGE
3. SOS (urgences)
4. ACCIDENT
5. CYBERCRIMINALITE
"""

import os
import sys
import json
import base64
from pathlib import Path
from PIL import Image
import io

# Importer Gemini et notre moteur
import google.generativeai as genai
from gemini_vision import GeminiVisionEngine

# Couleurs
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(title):
    print(f"\n{MAGENTA}{BOLD}{'='*70}{RESET}")
    print(f"{MAGENTA}{BOLD}{title:^70}{RESET}")
    print(f"{MAGENTA}{BOLD}{'='*70}{RESET}\n")

def print_module(module_name):
    print(f"{BLUE}{BOLD}📱 MODULE: {module_name.upper()}{RESET}")
    print(f"{BLUE}{'-'*50}{RESET}\n")

def create_test_image(color='red', description="test"):
    """Crée une image test avec couleur spécifique"""
    colors = {
        'red': 'red',
        'green': 'green',
        'brown': '#8B4513',
        'gray': 'gray',
        'white': 'white'
    }
    img = Image.new('RGB', (300, 300), color=colors.get(color, color))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_module(module_name, category, description=""):
    """Test un module spécifique"""
    print_module(module_name)
    
    try:
        # Initialiser le moteur Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        engine = GeminiVisionEngine(gemini_key)
        
        # Créer image test adaptée au module
        if category == "agriculture":
            img = create_test_image('green', 'farm')
            desc = "Champ avec cultures variées"
        elif category == "elevage":
            img = create_test_image('brown', 'animal')
            desc = "Vache ou animal au pâturage"
        elif category == "sos":
            img = create_test_image('red', 'emergency')
            desc = "Situation d'urgence potentielle"
        elif category == "accident":
            img = create_test_image('gray', 'accident')
            desc = "Scène d'accident"
        elif category == "cybercriminalite":
            img = create_test_image('white', 'cyber')
            desc = "Écran d'ordinateur ou mail suspect"
        else:
            img = create_test_image()
            desc = description
        
        # Encoder l'image
        img_b64 = base64.standard_b64encode(img.read()).decode()
        
        # Analyser avec Gemini
        print(f"{YELLOW}🔍 Analyse en cours...{RESET}\n")
        
        result = engine.analyze_images(
            images_data=[img_b64],
            text_description=desc,
            category=category
        )
        
        # Afficher les résultats
        print(f"{GREEN}✅ Analyse complète!{RESET}\n")
        print(f"{YELLOW}Résultats JSON:{RESET}")
        
        # Afficher les clés principales selon le module
        if category == "agriculture":
            print(f"  • Maladie détectée: {result.get('disease_detected', 'N/A')}")
            print(f"  • Confiance: {result.get('confidence', 0):.1%}")
            print(f"  • Urgence: {result.get('urgency', 'N/A')}")
            print(f"  • Traitement: {result.get('treatment', 'N/A')[:50]}...")
            
        elif category == "elevage":
            print(f"  • Animal affecté: {result.get('disease_detected', 'N/A')}")
            print(f"  • Confiance: {result.get('confidence', 0):.1%}")
            print(f"  • Urgence: {result.get('urgency', 'N/A')}")
            print(f"  • Soin recommandé: {result.get('treatment', 'N/A')[:50]}...")
            
        elif category == "sos":
            print(f"  • Urgence détectée: {result.get('emergency_detected', False)}")
            print(f"  • Gravité: {result.get('severity', 'N/A')}")
            print(f"  • Type de risque: {result.get('risk_type', 'N/A')}")
            print(f"  • Appels à faire: {result.get('call_services', 'N/A')}")
            actions = result.get('immediate_actions', [])
            if actions:
                print(f"  • Actions immédiates:")
                for action in actions[:2]:
                    print(f"    - {action}")
            
        elif category == "accident":
            print(f"  • Accident détecté: {result.get('accident_detected', False)}")
            print(f"  • Gravité: {result.get('severity', 'N/A')}")
            print(f"  • Victimes: {result.get('number_victims', 'N/A')}")
            print(f"  • Secours nécessaires: {', '.join(result.get('required_help', []))}")
            injuries = result.get('visible_injuries', [])
            if injuries:
                print(f"  • Blessures visibles:")
                for injury in injuries[:2]:
                    print(f"    - {injury}")
            
        elif category == "cybercriminalite":
            print(f"  • Menace détectée: {result.get('threat_detected', False)}")
            print(f"  • Type de menace: {result.get('threat_type', 'N/A')}")
            print(f"  • Niveau de risque: {result.get('risk_level', 'N/A')}")
            print(f"  • Confiance: {result.get('confidence', 0):.1%}")
            actions = result.get('recommended_actions', [])
            if actions:
                print(f"  • Actions recommandées:")
                for action in actions[:2]:
                    print(f"    - {action}")
        
        print(f"\n  • Modèle: {result.get('model', 'N/A')}")
        print(f"  • Analyse: {result.get('analysis', 'N/A')[:100]}...\n")
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ ERREUR: {str(e)}{RESET}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Teste tous les modules"""
    print_header("🔬 TEST GEMINI - TOUS LES MODULES RESOLVE HUB")
    
    # Vérifier l'API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print(f"{RED}❌ ERREUR: GEMINI_API_KEY non définie{RESET}\n")
        return 1
    
    print(f"{GREEN}✅ API Key détectée: {gemini_key[:15]}...{gemini_key[-5:]}{RESET}\n")
    
    # Tester les modules
    modules = [
        ("Agriculture", "agriculture"),
        ("Élevage", "elevage"),
        ("SOS - Urgence", "sos"),
        ("Accident", "accident"),
        ("Cybercriminalité", "cybercriminalite"),
    ]
    
    results = {}
    for module_name, category in modules:
        results[module_name] = test_module(module_name, category)
    
    # Résumé
    print_header("📊 RÉSUMÉ DES TESTS")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for module_name, success in results.items():
        status = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
        print(f"  {status} {module_name:25}")
    
    print(f"\n{BLUE}Score: {passed}/{total}{RESET}\n")
    
    if passed == total:
        print(f"{GREEN}{BOLD}🎉 Tous les modules testés avec succès!{RESET}")
        print(f"{GREEN}Le scanner est prêt à fonctionner avec les 5 modules.{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}{BOLD}⚠️ Certains modules ont échoué{RESET}")
        print(f"{RED}Vérifiez la configuration Gemini{RESET}\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
