#!/usr/bin/env python3
"""
Test complet du Scanner SONGRA
- Teste l'API backend FastAPI
- Simule l'appel frontend pour analyse photos
- Teste les 5 modules avec images
"""

import requests
import json
import base64
import io
from PIL import Image
import sys

# Couleurs
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Configuration
BACKEND_URL = "http://localhost:8000"
BACKEND_API = f"{BACKEND_URL}/api"

def print_header(title):
    print(f"\n{MAGENTA}{BOLD}{'='*70}{RESET}")
    print(f"{MAGENTA}{BOLD}{title:^70}{RESET}")
    print(f"{MAGENTA}{BOLD}{'='*70}{RESET}\n")

def print_test(name):
    print(f"{BLUE}{BOLD}[TEST] {name}{RESET}")

def check_backend():
    """Vérifie que le backend est accessible"""
    print_test("Vérification du backend FastAPI")
    try:
        response = requests.get(f"{BACKEND_URL}/docs", timeout=2)
        print(f"{GREEN}[OK] Backend accessible sur {BACKEND_URL}{RESET}\n")
        return True
    except Exception as e:
        print(f"{RED}[ERROR] Backend non accessible: {e}{RESET}")
        print(f"{YELLOW}Assurez-vous que le backend tourne: python run.py{RESET}\n")
        return False

def login_test():
    """Teste le login avec les credentials de test"""
    print_test("Login avec credentials de test")
    
    login_data = {
        "email": "test@resolvehub.bf",
        "password": "test123"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_API}/auth/login",
            json=login_data,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token") or data.get("token")
            if token:
                print(f"{GREEN}[OK] Login reussi{RESET}")
                print(f"    Token: {token[:30]}...{token[-10:] if len(token) > 40 else ''}")
                return token
            else:
                print(f"{RED}[ERROR] Pas de token dans la reponse{RESET}")
                print(f"    Response: {data}")
                return None
        else:
            print(f"{RED}[ERROR] Login failed: {response.status_code}{RESET}")
            print(f"    {response.text}")
            return None
    except Exception as e:
        print(f"{RED}[ERROR] {e}{RESET}\n")
        return None

def create_test_image(color='green', size=(300, 300)):
    """Crée une image PNG de test"""
    img = Image.new('RGB', size, color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return base64.standard_b64encode(img_bytes.read()).decode('utf-8')

def test_scanner_endpoint(token, category='agriculture'):
    """Teste l'endpoint /analyze_scanner_photo"""
    print_test(f"Test endpoint scanner - Module: {category.upper()}")
    
    # Créer image de test
    image_b64 = create_test_image('green' if category == 'agriculture' else 'brown')
    
    payload = {
        "images": [image_b64],
        "text_description": f"Image test pour module {category}",
        "category": category
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_API}/analyze_scanner_photo",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Afficher les résultats
            print(f"{GREEN}[OK] Reponse reçue{RESET}")
            
            # Afficher les clés principales selon le module
            print(f"\n    Results:")
            
            if category == 'agriculture':
                print(f"      • Maladie: {data.get('disease_detected', 'N/A')}")
                print(f"      • Confiance: {data.get('confidence', 0):.1%}")
                print(f"      • Urgence: {data.get('urgency', 'N/A')}")
                print(f"      • Modele: {data.get('model', 'N/A')}")
                
            elif category == 'elevage':
                print(f"      • Animal: {data.get('disease_detected', 'N/A')}")
                print(f"      • Confiance: {data.get('confidence', 0):.1%}")
                print(f"      • Urgence: {data.get('urgency', 'N/A')}")
                print(f"      • Modele: {data.get('model', 'N/A')}")
                
            elif category == 'sos':
                print(f"      • Urgence detectee: {data.get('emergency_detected', False)}")
                print(f"      • Gravite: {data.get('severity', 'N/A')}")
                print(f"      • Services: {data.get('call_services', 'N/A')}")
                print(f"      • Modele: {data.get('model', 'N/A')}")
                
            elif category == 'accident':
                print(f"      • Accident: {data.get('accident_detected', False)}")
                print(f"      • Gravite: {data.get('severity', 'N/A')}")
                print(f"      • Victimes: {data.get('number_victims', 'N/A')}")
                print(f"      • Modele: {data.get('model', 'N/A')}")
                
            elif category == 'cybercriminalite':
                print(f"      • Menace: {data.get('threat_detected', False)}")
                print(f"      • Type: {data.get('threat_type', 'N/A')}")
                print(f"      • Risque: {data.get('risk_level', 'N/A')}")
                print(f"      • Modele: {data.get('model', 'N/A')}")
            
            print(f"\n")
            return True
            
        else:
            print(f"{RED}[ERROR] Status {response.status_code}{RESET}")
            print(f"    {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"{RED}[ERROR] {e}{RESET}\n")
        return False

def test_knowledge_base(token):
    """Teste l'endpoint RAG de base de connaissances"""
    print_test("Test endpoint RAG - Base de connaissances")
    
    payload = {
        "question": "Comment traiter une carence en azote?",
        "domain": "agriculture"
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_API}/chat",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}[OK] RAG reponse reçue{RESET}")
            print(f"    Reponse: {data.get('response', 'N/A')[:150]}...")
            print(f"    Fiches trouvees: {len(data.get('rag_items', []))}")
            print(f"\n")
            return True
        else:
            print(f"{RED}[ERROR] Status {response.status_code}{RESET}")
            return False
            
    except Exception as e:
        print(f"{RED}[ERROR] {e}{RESET}\n")
        return False

def main():
    """Lance tous les tests"""
    print_header("🧪 TEST COMPLET DU SCANNER SONGRA")
    
    # 1. Vérifier backend
    if not check_backend():
        return 1
    
    # 2. Login
    token = login_test()
    if not token:
        print(f"{RED}Impossible de continuer sans token{RESET}\n")
        return 1
    
    print(f"\n")
    
    # 3. Tester les 5 modules du scanner
    modules = [
        "agriculture",
        "elevage",
        "sos",
        "accident",
        "cybercriminalite"
    ]
    
    results = {}
    for module in modules:
        results[module] = test_scanner_endpoint(token, module)
    
    # 4. Tester RAG
    results['rag'] = test_knowledge_base(token)
    
    # Résumé
    print_header("📊 RESULTAT FINAL")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"  {status} {name:20}")
    
    print(f"\n{BLUE}Score: {passed}/{total}{RESET}\n")
    
    if passed == total:
        print(f"{GREEN}{BOLD}🎉 SCANNER COMPLETEMENT OPERATIONNEL!{RESET}")
        print(f"{GREEN}L'application est prete pour la production.{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}{BOLD}⚠️ Certains tests ont echoue{RESET}")
        print(f"{YELLOW}Verifiez la configuration du backend{RESET}\n")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[INTERRUPTED] Test interrompu{RESET}\n")
        sys.exit(1)
