#!/usr/bin/env python3
"""
Test simplifié du Scanner - teste l'analyse Gemini directement
"""

import base64
import io
from PIL import Image
import sys
from gemini_vision import GeminiVisionEngine
import os

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

def create_test_image(color='green', label='test'):
    """Crée une image PNG de test"""
    img = Image.new('RGB', (300, 300), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return base64.standard_b64encode(img_bytes.read()).decode('utf-8')

def test_module(module_name, category, color='green'):
    """Test un module spécifique"""
    print(f"{BLUE}[TEST] Module: {module_name.upper()}{RESET}")
    
    try:
        # Initialiser Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            # Charger depuis .env
            from dotenv import load_dotenv
            load_dotenv()
            gemini_key = os.getenv("GEMINI_API_KEY")
        
        engine = GeminiVisionEngine(gemini_key)
        
        # Créer image et encoder
        image_b64 = create_test_image(color, module_name)
        
        # Analyser
        print(f"  [SEND] Envoi à Gemini...")
        result = engine.analyze_images(
            images_data=[image_b64],
            text_description=f"Image test pour {module_name}",
            category=category
        )
        
        # Afficher résultats
        print(f"{GREEN}[OK] Analyse reussie{RESET}")
        
        if category == 'agriculture':
            print(f"  • Maladie: {result.get('disease_detected', 'N/A')}")
            print(f"  • Confiance: {result.get('confidence', 0):.1%}")
        elif category == 'elevage':
            print(f"  • Animal: {result.get('disease_detected', 'N/A')}")
            print(f"  • Confiance: {result.get('confidence', 0):.1%}")
        elif category == '__sos':
            print(f"  • Urgence: {result.get('emergency_detected', False)}")
            print(f"  • Gravite: {result.get('severity', 'N/A')}")
        elif category == 'accident':
            print(f"  • Accident: {result.get('accident_detected', False)}")
            print(f"  • Gravite: {result.get('severity', 'N/A')}")
        elif category == 'cybercriminalite':
            print(f"  • Menace: {result.get('threat_detected', False)}")
            print(f"  • Type: {result.get('threat_type', 'N/A')}")
        
        print(f"  • Modele: {result.get('model', 'N/A')}")
        print(f"  • Analyse: {result.get('analysis', 'N/A')[:80]}...\n")
        
        return True
        
    except Exception as e:
        print(f"{RED}[ERROR] {str(e)}{RESET}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test rapide du scanner"""
    print_header("🧪 TEST SIMPLIFIE DU SCANNER - ANALYSIS PHOTO")
    
    modules = [
        ("Agriculture", "agriculture", "green"),
        ("Elevage", "elevage", "brown"),
        ("SOS", "sos", "red"),
        ("Accident", "accident", "gray"),
        ("Cybercriminalite", "cybercriminalite", "white"),
    ]
    
    results = {}
    for name, category, color in modules:
        results[name] = test_module(name, category, color)
    
    # Résumé
    print_header("📊 RESULTAT FINAL")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"  {status} {name:20}")
    
    print(f"\n{BLUE}Score: {passed}/{total}{RESET}\n")
    
    if passed == total:
        print(f"{GREEN}{BOLD}🎉 TOUS LES MODULES DU SCANNER FONCTIONNE!{RESET}")
        print(f"{GREEN}L'analse photo avec Gemini est operationnelle.{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}{BOLD}⚠️ Certains modules ont echoue{RESET}\n")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrompu{RESET}\n")
        sys.exit(1)
