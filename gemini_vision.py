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
                context_prompt = """TÂCHE: Analyser une photo agricole pour identifier les maladies des cultures.

INSTRUCTIONS IMPORTANTES:
- Analysez la photo avec ATTENTION aux détails
- Identifiez chaque culture visible
- Détectez TOUTE maladie, ravageur ou problème visible
- Si AUCUN problème: écrivez "Aucune maladie détectée"
- Donnez une confiance entre 0.0 (aucune certitude) et 1.0 (certitude totale)
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS:
{
    "disease_detected": "Maladie ou 'Aucune'",
    "confidence": 0.85,
    "symptoms": ["Symptôme 1", "Symptôme 2"],
    "treatment": "Action recommandée",
    "urgency": "low|medium|high",
    "prevents": "Prévention",
    "visual_observations": ["Détail observé"],
    "analysis": "Analyse détaillée en français"
}"""
            elif category == "elevage":
                context_prompt = """TÂCHE: Analyser une photo d'animal pour identifier les maladies et problèmes de santé.

INSTRUCTIONS IMPORTANTES:
- Analysez la photo avec ATTENTION
- Identifiez l'espèce et l'état de l'animal
- Détectez TOUTE maladie, blessure ou anomalie
- Si AUCUN problème visible: écrivez "Aucun"
- Donnez une confiance entre 0.0 et 1.0
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS:
{
    "disease_detected": "Maladie ou 'Aucun'",
    "confidence": 0.85,
    "symptoms": ["Symptôme visible"],
    "treatment": "Action de traitement/aide",
    "urgency": "low|medium|high",
    "prevents": "Prévention future",
    "visual_observations": ["Observation"],
    "analysis": "Analyse détaillée en français"
}"""
            elif category == "sos":
                context_prompt = """TÂCHE: Évaluer une situation d'urgence/SOS pour identifier les risques et actions immédiates.

INSTRUCTIONS CRITIQUES:
- Analysez la situation avec URGENCE
- Identifiez les risques immédiats (sécurité, santé, environnement)
- Déterminez le niveau de gravité: CRITIQUE, URGENT, MODÉRÉ
- Proposez actions immédiates (premiers secours, appels, évacuation)
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS:
{
    "emergency_detected": true/false,
    "severity": "critical|urgent|moderate|low",
    "risk_type": "Type de risque identifié",
    "confidence": 0.85,
    "immediate_actions": ["Action 1", "Action 2"],
    "call_services": "Police/Ambulance/Pompiers si nécessaire",
    "dangers": ["Danger 1", "Danger 2"],
    "first_aid": "Premiers secours à appliquer",
    "analysis": "Analyse détaillée en français"
}"""
            elif category == "accident":
                context_prompt = """TÂCHE: Analyser une photo d'accident pour évaluer les blessures et dégâts.

INSTRUCTIONS CRITIQUES:
- Analysez l'accident avec attention au détail
- Identifiez les blessures visibles (gravité, localisation)
- Évaluez les dégâts matériels
- Déterminez les secours nécessaires
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS:
{
    "accident_detected": true/false,
    "severity": "critical|serious|moderate|mild",
    "injury_types": ["Type de blessure"],
    "confidence": 0.85,
    "number_victims": "Nombre de victimes estimé",
    "urgency": "high|medium|low",
    "required_help": ["Ambulance", "Pompiers", "Police"],
    "visible_injuries": ["Blessure 1", "Blessure 2"],
    "damage_level": "Évaluation des dégâts",
    "analysis": "Analyse détaillée en français"
}"""
            elif category == "cybercriminalite":
                context_prompt = """TÂCHE: Analyser une image pour identifier les signes de cybercriminalité ou escroquerie.

INSTRUCTIONS IMPORTANTES:
- Analysez l'écran/document pour détecter arnaques, fraudes, malveillances
- Identifiez les indicateurs de risque (URLs suspectes, logos falsifiés, demandes manuelles données)
- Évaluez le niveau de dangerosité
- Proposez actions sécuritaires
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS:
{
    "threat_detected": true/false,
    "threat_type": "Phishing/Malware/Escroquerie/Faux site/Arnaque",
    "confidence": 0.85,
    "risk_level": "high|medium|low",
    "suspicious_elements": ["Élément 1", "Élément 2"],
    "recommended_actions": ["Ne pas cliquer", "Signaler", "Supprimer"],
    "safe_practice": "Recommandation de sécurité",
    "what_to_do": "Démarches à suivre",
    "analysis": "Analyse détaillée en français"
}"""
            else:
                context_prompt = """TÂCHE: Analyser cette image pour identifier tout problème de santé/maladie.

INSTRUCTIONS:
- Analysez soigneusement
- Identifiez problèmes visibles
- Donnez confiance 0.0 à 1.0
- RÉPONDEZ UNIQUEMENT EN JSON, PAS DE TEXTE SUPPLÉMENTAIRE

FORMAT JSON:
{
    "disease_detected": "Problème ou 'Aucun'",
    "confidence": 0.85,
    "symptoms": ["Symptôme"],
    "treatment": "Recommandation",
    "urgency": "low|medium|high",
    "prevents": "Prévention",
    "visual_observations": ["Observation"],
    "analysis": "Analyse en français"
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
