"""
Gemini Vision Engine pour analyse photo scanner
Remplace GPT-4o pour les photos (plus rapide, moins cher)
"""

import base64
import json
import re
from typing import List, Dict, Optional, Any
import google.generativeai as genai


class GeminiVisionEngine:
    """Analyse d'images via Google Gemini API (Vision)
    
    Utilisé UNIQUEMENT pour l'analyse de photos du scanner.
    Plus rapide et moins cher que GPT-4o.
    """
    
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("Gemini API key manquante")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print(f"[OK] Gemini model charge: {self.model.model_name}")
    
    def analyze_images(self, images_data: List[bytes], text_description: str = "", category: Optional[str] = None) -> Dict[str, Any]:
        """Analyser les images via Gemini Vision API"""
        valid_images = [image for image in images_data if image][:3]
        if not valid_images:
            raise ValueError("Aucune photo exploitable fournie")
        
        try:
            # Créer le prompt contextuel
            if category == "agriculture":
                context_prompt = """CONSULTATION EN LIGNE - DIAGNOSTIC AGRICOLE COMPLET - BURKINA FASO

TÂCHE: Analyser une photo agricole en 3 ÉTAPES OBLIGATOIRES:
  ÉTAPE 1 - OBSERVATION: Décrire précisément ce que vous voyez sur la photo
  ÉTAPE 2 - CARENCES & MALADIES: Détecter carences nutritionnelles, maladies, ravageurs
  ÉTAPE 3 - DIAGNOSTIC & TRAITEMENT: Diagnostic complet avec solutions adaptées

CONTEXTE BURKINABÈ:
- Climat sahélien, sécheresses périodiques
- Cultures: mil, sorgho, maïs, arachide, coton, niébé, sésame
- Ressources limitées, solutions locales prioritaires
- Accès aux experts parfois difficile

CARENCES À DÉTECTER (signes visuels clés):
- Carence Azote (N): feuilles jaunes depuis le bas, croissance ralentie, tiges fines
- Carence Phosphore (P): feuilles violacées/rougeâtres, racines peu développées
- Carence Potassium (K): bords feuilles brûlés/jaunis, fruits anormaux
- Carence Fer (Fe): jaunissement entre nervures (chlorose internervaire) sur jeunes feuilles
- Carence Magnésium (Mg): jaunissement entre nervures sur vieilles feuilles
- Carence Calcium (Ca): déformation jeunes feuilles, pourriture apicale
- Carence Zinc (Zn): petites feuilles, entre-nœuds courts, taches bronze
- Carence Soufre (S): jaunissement uniforme des jeunes feuilles
- Carence Manganèse (Mn): taches grises/beiges entre nervures
- Surhydratation ou Sécheresse: signes de stress hydrique

MALADIES & RAVAGEURS COURANTS BF:
- Mildiou, Rouille, Anthracnose, Fusariose
- Chenilles légionnaires, Pucerons, Acariens, Criquets
- Pourriture des racines, Fonte des semis

ANALYSE REQUISE:
1. Décrire précisément la plante et l'état du champ (couleur, forme, stade)
2. Lister TOUS les symptômes visibles (couleurs anormales, taches, déformations)
3. Identifier carence(s) probable(s) avec niveau de confiance
4. Identifier maladies/ravageurs présents
5. Évaluer gravité, urgence et impact sur récolte
6. Proposer traitement adapté au contexte BF (produits locaux + chimiques si nécessaire)
7. Recommander prévention future

IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, RIEN d'autre.

FORMAT JSON COMPLET:
{
    "consultation_type": "Diagnostic agricole complet",
    "what_i_see": "Description précise et détaillée de ce qui est visible sur la photo: type de plante, couleur des feuilles, état général, stade de croissance, conditions du champ, etc. C'est la première chose à dire avant tout diagnostic.",
    "culture_detected": "Maïs",
    "growth_stage": "Végétatif / Floraison / Fructification / Maturité",
    "general_plant_condition": "Bon / Moyen / Mauvais / Critique",
    "disease_detected": "Rouille foliacée ou 'Aucun problème détecté'",
    "deficiencies_detected": [
        {
            "deficiency": "Carence en Azote (N)",
            "confidence": 0.80,
            "symptoms_observed": ["Jaunissement des vieilles feuilles", "Feuilles jaunes en V depuis la pointe"],
            "impact": "Perte de rendement estimée 20-40% si non corrigée"
        }
    ],
    "all_symptoms": ["Taches oranges sur feuilles", "Jaunissement des bords", "Tige fine"],
    "confidence": 0.85,
    "severity": "modérée",
    "urgency": "medium",
    "diagnosis": "Explication complète et claire du ou des problèmes détectés: ce qui se passe dans la plante, pourquoi ces symptômes apparaissent, et les risques si non traité.",
    "treatment_steps": [
        "1. Action immédiate: ...",
        "2. Apporter engrais/traitement: ...",
        "3. Surveiller l'évolution",
        "4. Fréquence d'application"
    ],
    "deficiency_corrections": [
        "Pour carence Azote: urée (46-0-0) ou compost bien décomposé, 50kg/ha",
        "Pour carence Fer: sulfate ferreux en pulvérisation foliaire si disponible"
    ],
    "local_remedies": ["Compost de déchets ménagers", "Cendre de bois (potasse)", "Urine fermentée (azote)", "Fumier de parc"],
    "chemical_treatments": ["Engrais NPK disponible au marché local", "Fongicide cuivrique si maladie fongique"],
    "when_to_call_expert": "Si progression rapide ou plus de 50% de la récolte affectée",
    "prevention": "Rotation des cultures, amendement organique, semences certifiées, gestion de l'eau",
    "visual_observations": ["Couleur réelle des feuilles", "Zones affectées précises", "Stade de la plante"],
    "harvest_risk": "Risque faible / modéré / élevé sur la récolte actuelle",
    "analysis": "Résumé complet en français simple: Ce que j'observe → Ce que cela signifie → Ce qu'il faut faire MAINTENANT → Ce qu'il faut faire dans 1-2 semaines. Adapté à un agriculteur burkinabè sans formation spécialisée."
}"""
            elif category == "elevage":
                context_prompt = """CONSULTATION EN LIGNE - DIAGNOSTIC VÉTÉRINAIRE COMPLET - BURKINA FASO

TÂCHE: Analyser photo animal en 3 ÉTAPES OBLIGATOIRES:
  ÉTAPE 1 - OBSERVATION: Décrire précisément ce que vous voyez (espèce, état, anomalies visibles)
  ÉTAPE 2 - SYMPTÔMES: Identifier chaque signe anormal et sa signification médicale
  ÉTAPE 3 - DIAGNOSTIC & TRAITEMENT: Diagnostic complet avec solutions adaptées

CONTEXTE BURKINABÈ:
- Élevages: bovins, ovins, caprins, volailles adaptés climat sahélien
- Ressources vétérinaires limitées, priorité aux soins locaux
- Climat chaud, eau parfois insuffisante
- Maladies courantes: parasites, infections, malnutrition
- Nombreux petits éleveurs sans accès direct aux vétérinaires

ANALYSE REQUISE:
1. Décrire l'animal (espèce, âge apparent, état corporel, comportement visible)
2. Détecter CHAQUE signe de maladie/blessure/anomalie avec localisation précise
3. Évaluer gravité (critique, sérieux, modéré, léger)
4. Proposer traitement adapté ressources locales
5. Indiquer quand appeler un vétérinaire

IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, RIEN d'autre.

FORMAT JSON COMPLET (exemple):
{
    "consultation_type": "Diagnostic vétérinaire",
    "what_i_see": "Description précise et détaillée de l'animal sur la photo: espèce, taille, couleur du pelage/plumage, état général visible, posture, zones anormales observées, conditions environnement.",
    "animal_species": "Chèvre",
    "animal_condition": "Description générale: âge apparent, état corporel, comportement",
    "disease_detected": "Gale ou 'Aucun problème visible'",
    "confidence": 0.85,
    "severity": "modérée",
    "visible_symptoms": ["Perte de poils par plaques", "Rougeur cutanée", "Démangeaisons visibles"],
    "diagnosis": "Description médicale complète et claire du problème - adapté pour non-experts",
    "treatment_steps": [
        "1. Isoler l'animal des autres (si contagieux)",
        "2. Nettoyer avec eau douce et savon",
        "3. Appliquer huile de neem ou goudron local",
        "4. Traiter 2x par semaine pendant 3 semaines",
        "5. Nettoyer l'enclos et les outils quotidiennement"
    ],
    "local_remedies": ["Huile de neem", "Goudron de bois", "Sel + eau douce"],
    "medication_options": [
        "Si accès: Ivermectine ou Abamectine pour parasites externes",
        "Sinon: solutions naturelles locales"
    ],
    "when_to_call_vet": "Si après 2 semaines d'amélioration ou animal ne mange plus ou fièvre présente ou propagation à d'autres animaux",
    "urgency": "medium_not_critical",
    "prevention": "Hygiène enclos, eau propre quotidienne, alimentation équilibrée, rotation pâturages",
    "visual_observations": ["Zone affectée: cuisses et flancs", "Peau suintante légèrement"],
    "risk_spread": "Contagieux entre animaux du même groupe",
    "nutrition_impact": "À ce stade: impact modéré sur appétit",
    "analysis": "Explication complète et simple de ce qui se passe avec l'animal et pourquoi, adapté agriculteur Burkina Faso"
}"""
            elif category == "sos_accident":
                context_prompt = """CONSULTATION D'URGENCE - SOS ACCIDENT/PREMIERS SECOURS - BURKINA FASO

TÂCHE CRITIQUE: Évaluer l'urgence et recommander actions immédiates pour situation accident/grave.

CONTEXTE BURKINABÈ:
- Accès ambulance/hôpital parfois éloigné (heures)
- Les premiers secours LOCAUX sont cruciaux (premiers 30-60 min)
- Ressources médicales limitées
- Risques: blessures, infections, saignements graves

CLI: SI GRAVITÉ CRITIQUE → Actions IMMÉDIATES = PRIORITÉ

ANALYSE REQUISE:
1. Évaluer gravité IMMÉDIATE de la blessure/situation
2. Identifier signes de danger critique (saignement grave, inconscience, etc.)
3. Déterminer actions d'urgence MAINTENANT (avant ambulance)
4. Indiquer quand appeler services d'urgence (18 ou 15 Burkina Faso)

IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, RIEN d'autre.

FORMAT JSON COMPLET (exemple):
{
    "consultation_type": "SOS Premiers secours",
    "situation_type": "Plaie infectée avec rougeur et gonflement",
    "emergency_detected": true,
    "severity_level": "High_NOT_Critical_but_urgent",
    "critical_signs": ["Rougeur s'élargit", "Gonflement s'aggrave"],
    "danger_if_ignored": "Risque infection grave (septicémie) si non traité 1-2 jours",
    "immediate_actions": [
        "🚨 MAINTENANT - NE TOUCHEZ PAS:",
        "1. Laver à l'eau propre/bouillie (refroidir si brûlure)",
        "2. Sécher avec textile PROPRE",
        "3. Appliquer désinfectant local (Bétadine si disponible, sinon alcool 70°)",
        "4. Couvrir avec bande propre",
        "5. Donner paracétamol si douleur"
    ],
    "required_services": ["Consultation médicale dans 1-2 jours", "Appeler ambulance SI: fièvre>38.5°C ou gonflement galopant"],
    "call_emergency_if": [
        "Fièvre >38.5°C",
        "Gonflement du membre augmente de >2cm/jour",
        "Apparition de pus ou mauvaise odeur (infection avancée)",
        "Patient perd connaissance ou fièvre vomissements"
    ],
    "first_aid_details": "Étapes précises et réalistes sans matériel médical avancé",
    "pain_management": "Paracétamol 500mg x3/jour ou aspirine ajustée poids",
    "infection_monitoring": [ "Vérifier rougeur chaque 12h", "Mesurer gonflement", "Vérifier température"],
    "medications_to_avoid": ["Aucun antibiotique sans ordonnance"],
    "when_this_is_critical": "Si signes d'infection systémique (fièvre + frissons + malaise général)",
    "healing_timeline": "Si traitement immédiat correct: amélioration visible 3-5 jours",
    "prevention_infection": [
        "Garder bande DRY et PROPRE",
        "Changer bande 1-2x/jour",
        "Lavage doux quotidien à eau savonneuse",
        "Aucun contact avec animaux"
    ],
    "visual_observations": ["Plaie 2cm x 1.5cm", "Rougeur 4cm autour", "Gonflement léger"],
    "urgency_score": 7.5,
    "analysis": "Description claire du danger et de pourquoi c'est urgent, adapté pour personne sans formation médicale au Burkina Faso"
}"""
            elif category == "cybersecurity":
                context_prompt = """CONSULTATION EN LIGNE - CYBERSÉCURITÉ/ARNAQUE - BURKINA FASO

TÂCHE: Analyser image pour identifier arnaques, phishing, escroqueries, menaces en ligne.

CONTEXTE BURKINABÈ:
- Croissance mobile money et transactions digitales
- Arnaques ciblant utilisateurs inexpérimentés
- Messages WhatsApp/SMS frauduleux courants
- Usurpation identité et faux sites en augmentation

ANALYSE REQUISE:
1. Identifier tout indicateur d'arnaque/malveillance
2. Évaluer niveau de risque et perte potentielle
3. Expliquer les RED FLAGS clairement
4. Recommander actions sécuritaires

IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, RIEN d'autre.

FORMAT JSON:
{
    "consultation_type": "Cybersécurité - Arnaque détection",
    "threat_detected": true/false,
    "threat_type": "Phishing/Arnaque mobile money/Faux site/Malware/Autre",
    "confidence": 0.85,
    "risk_level": "critical|high|medium|low",
    "red_flags": ["Drapeau rouge 1", "Drapeau rouge 2"],
    "what_scammers_want": "Vos codes PIN / Mot de passe / Argent Mobile Money / Données personnelles",
    "immediate_actions": [
        "1. NE cliquez PAS sur le lien",
        "2. NE envoyez PAS codes ou informations",
        "3. Fermez/supprimez le message",
        "4. Changez votre mot de passe immédiatement"
    ],
    "how_to_verify_legitimate": "Comment savoir si c'est RÉELLEMENT de votre banque/plateforme",
    "safe_practices": ["Jamais partager code", "Appeler directement la banque"],
    "if_compromised": "Si vous avez déjà envoyé info, actions à faire",
    "reporting": "Où signaler l'arnaque (à qui, comment)",
    "analysis": "Explication simple du danger et pourquoi c'est une arnaque"
}"""
            else:
                context_prompt = """TÂCHE: Analyser cette image pour identifier tout problème visible.

INSTRUCTIONS:
- D'abord décrire ce que vous voyez (étape observation)
- Ensuite identifier les problèmes
- Donnez confiance 0.0 à 1.0
- RÉPONDEZ UNIQUEMENT EN JSON, PAS DE TEXTE SUPPLÉMENTAIRE

FORMAT JSON:
{
    "what_i_see": "Description précise de ce qui est visible sur la photo avant tout diagnostic",
    "disease_detected": "Problème ou 'Aucun'",
    "confidence": 0.85,
    "symptoms": ["Symptôme visible"],
    "treatment": "Recommandation concrète",
    "urgency": "low|medium|high",
    "prevention": "Prévention",
    "visual_observations": ["Observation détaillée 1", "Observation détaillée 2"],
    "analysis": "Résumé: Ce que je vois → Ce que cela signifie → Ce qu'il faut faire"
}"""
            
            # Construire le contenu du message
            prompt_text = context_prompt + (f"\n\nContext supplémentaire: {text_description}" if text_description else "")
            
            # Préparer les images pour Gemini
            content_parts = [prompt_text]
            
            for idx, img_bytes in enumerate(valid_images):
                print(f"[IMG] Image {idx + 1}: {len(img_bytes)} bytes ({len(img_bytes)/1024:.1f}KB) - {category} mode")
                # Gemini accepte base64 encode
                # Gerer les deux cas: bytes bruts ou string base64 deja encodee
                if isinstance(img_bytes, str):
                    img_b64 = img_bytes
                else:
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                content_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': img_b64
                })
            
            print(f"[SEND] Envoi a Gemini: {len(content_parts)-1} image(s) + prompt")
            
            # Appeler Gemini
            response = self.model.generate_content(content_parts)
            response_text = response.text
            
            print(f"[OK] Reponse Gemini recue")
            print(f"[LOG] Reponse brute Gemini: {response_text[:300]}...")
            
            # Parser la réponse
            analysis_json = None
            try:
                # Chercher le JSON dans la réponse
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    analysis_json = json.loads(json_str)
                    print(f"[OK] JSON parse: {analysis_json.get('disease_detected', 'N/A')}")
                else:
                    print(f"[WARN] Pas de JSON trouve dans reponse Gemini")
                    
            except json.JSONDecodeError as je:
                print(f"[WARN] JSON parsing error: {je}")
                # Essayer de nettoyer et re-parser
                try:
                    cleaned = response_text.replace('\n', ' ').replace('  ', ' ')
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        analysis_json = json.loads(json_match.group())
                        print(f"[OK] JSON parse apres nettoyage: {analysis_json.get('disease_detected', 'N/A')}")
                except:
                    print(f"[ERROR] Impossible parser JSON Gemini apres nettoyage")
            
            # Si toujours pas de JSON valide
            if not analysis_json:
                disease_keywords = ['maladie', 'malade', 'blessure', 'infection', 'aucun', 'aucune', 'normal', 'sain']
                detected = 'Non identifiée'
                for keyword in disease_keywords:
                    if keyword in response_text.lower():
                        detected = 'Détecté' if keyword not in ['aucun', 'aucune', 'normal', 'sain'] else 'Aucune maladie'
                        break
                
                print(f"[WARN] Utilisant reponse par defaut Gemini avec keyword matching")
                analysis_json = {
                    "disease_detected": detected,
                    "confidence": 0.4,
                    "analysis": response_text[:300] if response_text else "Analyse incomplète",
                    "urgency": "medium",
                    "requires_expert": False
                }
            
            # Enrichir avec métadonnées
            analysis_json["photo_count"] = len(valid_images)
            analysis_json["best_view_index"] = 1
            analysis_json["analyzed_views"] = [
                {
                    "view_index": i + 1,
                    "disease_detected": analysis_json.get("disease_detected"),
                    "confidence": analysis_json.get("confidence", 0.5)
                }
                for i in range(len(valid_images))
            ]
            analysis_json["requires_expert"] = analysis_json.get("urgency") == "high" or analysis_json.get("confidence", 0.5) < 0.6
            analysis_json["model"] = "gemini-2.5-flash"
            
            return analysis_json
            
        except Exception as e:
            print(f"❌ Erreur Gemini Vision: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "disease_detected": "Erreur analyse",
                "confidence": 0,
                "analysis": f"Erreur lors de l'analyse Gemini: {str(e)}",
                "urgency": "medium",
                "requires_expert": True,
                "photo_count": len(valid_images),
                "error": str(e),
                "model": "gemini-1.5-flash"
            }
