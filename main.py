"""
SONGRA - Backend API avec Computer Vision LOCALE
Version FINALE - Avec analyse IA complète
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import jwt
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import hashlib
import unicodedata
import json
import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from gemini_vision import GeminiVisionEngine

# ==========================================
# CONFIGURATION
# ==========================================

# Charger les variables d'environnement depuis un fichier .env (dev / prod)
# override=True permet de remplacer une éventuelle variable OPENAI_API_KEY déjà
# définie dans l'environnement Windows (ancienne clé) par la valeur du .env.
load_dotenv(override=True)

os.makedirs("uploads", exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resolvehub.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Gemini API Key pour analyse photo
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"[OK] Gemini API configuree")
else:
    print(f"[WARN] GEMINI_API_KEY non definie - scanner photo avec OpenAI")

JWT_SECRET = os.getenv("JWT_SECRET", "songra-mobile-dev-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

# ==========================================
# MODÈLES (Compatibles avec base existante)
# ==========================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=True)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_anonymized = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    premium_expires_at = Column(DateTime, nullable=True)
    messages_used = Column(Integer, default=0)
    messages_limit = Column(Integer, default=1)  # 1 gratuit, 10 pour premium

class Expert(Base):
    __tablename__ = "experts"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    specialization = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    expert_id = Column(Integer, nullable=True)
    category = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    status = Column(String, default="open")
    ai_confidence_score = Column(Float, nullable=True)
    ai_extracted_keywords = Column(String, nullable=True)
    ai_photo_analysis = Column(Text, nullable=True)
    photo_path = Column(String, nullable=True)  # NOM ORIGINAL - NE PAS CHANGER
    photo_paths_json = Column(Text, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class PhotoAnalysisHistoryDB(Base):
    __tablename__ = "photo_analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    client_record_id = Column(String, nullable=True)
    category = Column(String, nullable=True)
    prompt = Column(Text, nullable=True)
    analysis_json = Column(Text, nullable=True)
    photo_paths_json = Column(Text, nullable=True)
    photo_labels_json = Column(Text, nullable=True)
    source_ticket_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, nullable=False)
    sender_type = Column(String, nullable=False)
    sender_id = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    channel = Column(String, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, nullable=False)  # agriculture / elevage / cybersecurity / health
    title = Column(String, nullable=False)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    tags = Column(String, nullable=True)  # JSON list of tags
    language = Column(String, default="fr")
    source = Column(String, nullable=True)
    media = Column(Text, nullable=True)  # JSON list of media items (images / videos)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class EmergencyNumber(Base):
    __tablename__ = "emergency_numbers"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)  # Ex: Sapeurs-pompiers
    number = Column(String, nullable=False)  # Ex: 18, 112, etc.
    description = Column(Text, nullable=True)  # Note contextuelle (pays / région)
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)


Base.metadata.create_all(bind=engine)


def _ensure_user_auth_columns() -> None:
    """Ajouter les colonnes d'auth mobile si elles sont absentes."""
    if not str(engine.url).startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA table_info(users)")
            columns = [row[1] for row in result]

            if "password_hash" not in columns:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN password_hash TEXT")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter les colonnes d'auth utilisateur: {e}")


_ensure_user_auth_columns()


def _ensure_media_column_for_knowledge_items() -> None:
    """S'assurer que la colonne 'media' existe dans la table knowledge_items.

    Utile quand la base SQLite existait avant l'ajout de ce champ : on ajoute
    simplement la colonne manquante sans casser les données existantes.
    """
    # Fonction principalement pensée pour SQLite (dev/local). Pour d'autres
    # SGBD avec migrations gérées, on ne fait rien ici.
    if not str(engine.url).startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            # Récupérer la liste des colonnes existantes
            result = conn.exec_driver_sql("PRAGMA table_info(knowledge_items)")
            columns = [row[1] for row in result]  # row[1] = nom de colonne

            if "media" not in columns:
                conn.exec_driver_sql("ALTER TABLE knowledge_items ADD COLUMN media TEXT")
    except Exception as e:
        # On loggue mais on ne bloque pas le démarrage de l'API
        print(f"⚠️ Impossible d'ajouter la colonne 'media' à knowledge_items: {e}")


_ensure_media_column_for_knowledge_items()


def _ensure_ticket_photo_columns() -> None:
    if not str(engine.url).startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA table_info(tickets)")
            columns = [row[1] for row in result]

            if "photo_paths_json" not in columns:
                conn.exec_driver_sql("ALTER TABLE tickets ADD COLUMN photo_paths_json TEXT")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter la colonne 'photo_paths_json' à tickets: {e}")


_ensure_ticket_photo_columns()

# ==========================================
# MODÈLES PYDANTIC
# ==========================================

class ConversationTurn(BaseModel):
    role: str
    content: str

class MessageCreate(BaseModel):
    content: str
    phone_number: str
    channel: str = "app"
    category: Optional[str] = None  # catégorie choisie côté app (agriculture, elevage, sos_accident, cybersecurity)
    photo_base64: Optional[str] = None
    photo_base64_list: Optional[List[str]] = None
    conversation_context: Optional[List[ConversationTurn]] = None

class ExpertLogin(BaseModel):
    email: str
    password: str


class UserRegister(BaseModel):
    phone_number: str
    password: str
    name: str
    location: Optional[str] = None


class UserLogin(BaseModel):
    phone_number: str
    password: str


class MobileQuestionCreate(BaseModel):
    content: str
    category: Optional[str] = None
    photo_base64: Optional[str] = None
    photo_base64_list: Optional[List[str]] = None
    conversation_context: Optional[List[ConversationTurn]] = None

class ReplyMessage(BaseModel):
    message: str


class PhotoAnalysisHistoryIn(BaseModel):
    phone_number: str
    client_record_id: Optional[str] = None
    category: Optional[str] = None
    prompt: Optional[str] = None
    analysis: Dict[str, Any]
    photo_base64_list: List[str] = []
    photo_labels: List[str] = []


class KnowledgeMedia(BaseModel):
    type: str
    url: str
    title: Optional[str] = None


class KnowledgeItemIn(BaseModel):
    domain: str
    title: str
    question: Optional[str] = None
    answer: str
    tags: List[str] = []
    language: str = "fr"
    source: Optional[str] = None
    media: Optional[List[KnowledgeMedia]] = None


class KnowledgeBulkImport(BaseModel):
    items: List[KnowledgeItemIn]


class EmergencyNumberIn(BaseModel):
    label: str
    number: str
    description: Optional[str] = None
    display_order: int = 0


# ==========================================
# MODULE IA PHOTO LOCALE (Computer Vision) - RESTAURÉ
# ==========================================

class LocalComputerVision:
    """
    Système de Computer Vision 100% LOCAL
    Détection de maladies des plantes sans API externe
    """
    
    def __init__(self):
        # Base de connaissances des maladies courantes au Burkina Faso
        self.diseases_database = {
            "mais_taches_jaunes": {
                "name": "Carence en Azote",
                "confidence_keywords": ["jaune", "feuille", "maïs", "sécher"],
                "symptoms": ["Jaunissement des feuilles du bas vers le haut", "Croissance ralentie"],
                "treatment": "Appliquer engrais NPK (10-10-10) à 50kg/ha. Améliorer drainage. Arrosage régulier matin/soir.",
                "urgency": "medium",
                "prevention": "Rotation des cultures, compost organique, analyse sol annuelle"
            },
            "mais_rouille": {
                "name": "Rouille du Maïs",
                "confidence_keywords": ["tache", "orange", "rouille", "poudre"],
                "symptoms": ["Pustules orange/brunes sur feuilles", "Aspect poudreuse"],
                "treatment": "Fongicide naturel (purin d'ortie dilué 1:10). Retirer feuilles infectées. Espacer plants.",
                "urgency": "high",
                "prevention": "Variétés résistantes, rotation, bon espacement"
            },
            "tomate_mildiou": {
                "name": "Mildiou de la Tomate",
                "confidence_keywords": ["tache", "brun", "noir", "tomate", "pourrir"],
                "symptoms": ["Taches brunes/noires sur feuilles", "Fruits pourrissent"],
                "treatment": "URGENT: Retirer plants infectés. Bouillie bordelaise. Éviter arrosage feuilles.",
                "urgency": "high",
                "prevention": "Paillage, arrosage au pied, aération"
            },
            "sorgho_charbon": {
                "name": "Charbon du Sorgho",
                "confidence_keywords": ["noir", "poudre", "épi", "sorgho"],
                "symptoms": ["Masse noire poudreuse remplace grains"],
                "treatment": "Détruire plants infectés (brûler). Traiter semences. Rotation 3 ans.",
                "urgency": "high",
                "prevention": "Semences certifiées traitées, rotation cultures"
            },
            "manioc_mosaique": {
                "name": "Mosaïque du Manioc",
                "confidence_keywords": ["mosaïque", "déformation", "feuille", "manioc"],
                "symptoms": ["Motif mosaïque jaune/vert sur feuilles", "Déformation"],
                "treatment": "Pas de traitement. Arracher et détruire. Utiliser boutures saines certifiées.",
                "urgency": "high",
                "prevention": "Boutures certifiées, contrôle pucerons, éliminer plants malades"
            },
            "animal_fievre": {
                "name": "Fièvre Animale (suspicion)",
                "confidence_keywords": ["bétail", "fièvre", "faible", "animal"],
                "symptoms": ["Température élevée", "Perte appétit", "Faiblesse"],
                "treatment": "CONSULTER vétérinaire RAPIDEMENT. Isoler animal. Eau fraîche disponible.",
                "urgency": "high",
                "prevention": "Vaccination, vermifugation, abri ombragé"
            },
            "animal_plaie": {
                "name": "Plaie ou traumatisme animal",
                "confidence_keywords": ["plaie", "blessure", "saigne", "boite", "coupure", "peau"],
                "symptoms": ["Rougeur, saignement ou lésion visible", "Douleur ou gêne au déplacement"],
                "treatment": "Nettoyer la zone avec une solution antiseptique adaptée, limiter les mouches et isoler l'animal si nécessaire.",
                "urgency": "high",
                "prevention": "Inspecter les clôtures, retirer les objets coupants et surveiller les parasites.",
            },
            "animal_infection_cutanee": {
                "name": "Infection cutanée animale (suspicion)",
                "confidence_keywords": ["croûte", "peau", "purulent", "plaque", "démangeaison"],
                "symptoms": ["Croûtes, inflammation ou dépilation locale", "Zone cutanée anormale persistante"],
                "treatment": "Nettoyer la zone, éviter le léchage ou frottement et demander un avis vétérinaire pour confirmer le traitement.",
                "urgency": "medium",
                "prevention": "Hygiène de l'abri, contrôle des parasites et isolement des animaux atteints.",
            },
            "animal_oculaire": {
                "name": "Atteinte oculaire ou nasale animale",
                "confidence_keywords": ["oeil", "œil", "ecoulement", "nez", "narine", "crête"],
                "symptoms": ["Écoulement, irritation ou gonflement visible", "Atteinte possible des voies respiratoires ou des yeux"],
                "treatment": "Isoler l'animal, nettoyer délicatement les sécrétions externes et consulter rapidement un agent d'élevage ou vétérinaire.",
                "urgency": "high",
                "prevention": "Ventilation correcte, réduction de la promiscuité et surveillance du lot.",
            },
            "human_plaie_hemorragique": {
                "name": "Plaie ouverte ou saignement",
                "confidence_keywords": ["sang", "saigne", "plaie", "coupure", "blessure", "accident"],
                "symptoms": ["Plaie ouverte ou saignement visible", "Atteinte cutanée nécessitant compression ou pansement"],
                "treatment": "Comprimer la plaie avec un tissu propre, surélever si possible la zone touchée et consulter d'urgence si le saignement persiste.",
                "urgency": "high",
                "prevention": "Port de protections et désinfection rapide des petites coupures.",
            },
            "human_brule": {
                "name": "Brûlure ou irritation thermique",
                "confidence_keywords": ["brulure", "brûlure", "chaud", "huile", "feu", "peau rouge"],
                "symptoms": ["Rougeur diffuse, cloque ou surface brûlée", "Douleur et inflammation locale"],
                "treatment": "Refroidir immédiatement à l'eau propre tempérée pendant 10 à 20 minutes. Ne pas appliquer de produit agressif ni percer les cloques.",
                "urgency": "high",
                "prevention": "Manipuler chaleur et liquides bouillants avec protection adaptée.",
            },
            "human_infection_plaie": {
                "name": "Plaie infectée ou inflammatoire",
                "confidence_keywords": ["pus", "infecte", "infectée", "gonfle", "rouge", "chaud"],
                "symptoms": ["Rougeur persistante, gonflement ou écoulement", "Suspicion d'infection locale"],
                "treatment": "Nettoyer la plaie à l'eau propre, couvrir avec un pansement propre et consulter un soignant si douleur, fièvre ou pus apparaissent.",
                "urgency": "high",
                "prevention": "Désinfecter tôt les plaies et renouveler les pansements propres.",
            },
            "human_contusion": {
                "name": "Contusion ou hématome",
                "confidence_keywords": ["choc", "tombe", "bleu", "gonfle", "douleur", "coup"],
                "symptoms": ["Coloration sombre ou tuméfaction visible", "Douleur localisée après choc"],
                "treatment": "Appliquer du froid enveloppé, surélever la zone si possible et surveiller douleur intense ou incapacité à bouger.",
                "urgency": "medium",
                "prevention": "Protéger les zones exposées et sécuriser les zones de travail ou de déplacement.",
            },
            "mais_helminthosporiose": {
                "name": "Helminthosporiose / brûlure foliaire du maïs",
                "confidence_keywords": ["mais", "maïs", "taches allongees", "brun", "feuille", "brule"],
                "symptoms": ["Taches allongées brunes sur feuilles", "Dessèchement progressif du feuillage"],
                "treatment": "Retirer les feuilles très atteintes, améliorer l'aération et utiliser un traitement fongique adapté si disponible localement.",
                "urgency": "high",
                "prevention": "Rotation culturale, destruction des résidus malades et semences saines.",
            },
            "manioc_bacteriose": {
                "name": "Brûlure bactérienne du manioc (suspicion)",
                "confidence_keywords": ["manioc", "brulure", "brûlure", "feuille", "dessèchement", "bacteriose"],
                "symptoms": ["Brunissement et dessèchement foliaire", "Atteinte progressive des feuilles ou tiges"],
                "treatment": "Éliminer les plants très atteints, désinfecter les outils et éviter les boutures issues des plants suspects.",
                "urgency": "high",
                "prevention": "Utiliser des boutures saines, éviter la propagation mécanique et pratiquer la rotation.",
            },
            "animal_pied_lesion": {
                "name": "Lésion du pied ou du sabot",
                "confidence_keywords": ["pied", "sabot", "boiterie", "boite", "patte", "plaie"],
                "symptoms": ["Boiterie ou douleur à l'appui", "Lésion visible au pied ou au sabot"],
                "treatment": "Nettoyer le pied, limiter les déplacements et faire vérifier rapidement si l'animal ne pose plus correctement le membre.",
                "urgency": "high",
                "prevention": "Assainir les sols humides, inspecter régulièrement les sabots et retirer les objets blessants.",
            },
            "volaille_variole": {
                "name": "Variole aviaire (suspicion)",
                "confidence_keywords": ["volaille", "croute", "crete", "crête", "face", "bouton"],
                "symptoms": ["Croûtes ou nodules sur tête, crête ou autour des yeux", "Atteinte cutanée évocatrice chez la volaille"],
                "treatment": "Isoler la volaille, désinfecter l'abri et consulter rapidement un technicien d'élevage pour confirmer la conduite à tenir.",
                "urgency": "high",
                "prevention": "Lutter contre les moustiques, isoler les sujets atteints et renforcer l'hygiène du poulailler.",
            }
        }
        
        # Maladies par culture pour reconnaissance rapide
        self.crop_diseases = {
            "maïs": ["mais_taches_jaunes", "mais_rouille"],
            "tomate": ["tomate_mildiou"],
            "sorgho": ["sorgho_charbon"],
            "manioc": ["manioc_mosaïque"],
            "bétail": ["animal_fievre"]
        }
        self.subject_profiles = {
            "mais": {
                "label": "Maïs",
                "keywords": ["mais", "maïs", "epi", "épi"],
                "capture_guidance": [
                    "Vue générale de la parcelle ou du plant",
                    "Gros plan des feuilles jaunies ou tachées",
                    "Photo du revers de la feuille ou de l'épi",
                ],
            },
            "tomate": {
                "label": "Tomate",
                "keywords": ["tomate", "fruit", "tige"],
                "capture_guidance": [
                    "Vue générale du plant de tomate",
                    "Gros plan des feuilles touchées",
                    "Photo des fruits ou de la tige atteinte",
                ],
            },
            "manioc": {
                "label": "Manioc",
                "keywords": ["manioc", "bouture"],
                "capture_guidance": [
                    "Vue générale du plant de manioc",
                    "Gros plan d'une feuille entière",
                    "Photo du revers des feuilles ou des jeunes pousses",
                ],
            },
            "sorgho": {
                "label": "Sorgho",
                "keywords": ["sorgho", "panicule", "epi", "épi"],
                "capture_guidance": [
                    "Vue générale du plant de sorgho",
                    "Photo de l'épi ou panicule",
                    "Gros plan de la zone noircie ou poudreuse",
                ],
            },
            "oignon": {
                "label": "Oignon",
                "keywords": ["oignon", "bulbe"],
                "capture_guidance": [
                    "Vue générale du rang d'oignons",
                    "Gros plan des feuilles ou du collet",
                    "Photo du bulbe si possible",
                ],
            },
            "arachide": {
                "label": "Arachide",
                "keywords": ["arachide", "cacahuete", "cacahuète"],
                "capture_guidance": [
                    "Vue générale du plant d'arachide",
                    "Gros plan des folioles tachées",
                    "Photo du pied et du sol autour",
                ],
            },
            "betail": {
                "label": "Bétail",
                "keywords": ["betail", "bétail", "vache", "boeuf", "bovin", "veau"],
                "capture_guidance": [
                    "Vue générale de l'animal",
                    "Gros plan de la zone touchée",
                    "Photo des yeux, de la bouche ou du museau si anormal",
                ],
            },
            "petit_ruminant": {
                "label": "Petit ruminant",
                "keywords": ["chevre", "chèvre", "mouton", "brebis"],
                "capture_guidance": [
                    "Vue générale de l'animal",
                    "Gros plan de la peau ou de la lésion",
                    "Photo des yeux ou de la bouche si écoulement",
                ],
            },
            "volaille": {
                "label": "Volaille",
                "keywords": ["volaille", "poule", "coq", "poulet", "canard"],
                "capture_guidance": [
                    "Vue générale de la volaille",
                    "Gros plan de la tête, des yeux ou de la crête",
                    "Photo de la zone plumage ou peau touchée",
                ],
            },
            "lapin": {
                "label": "Lapin",
                "keywords": ["lapin", "lapins", "lapereau", "lapereaux", "clapier"],
                "capture_guidance": [
                    "Vue générale du lapin dans le clapier",
                    "Gros plan de la zone touchée ou de la tête",
                    "Photo des yeux, du nez, des oreilles ou des pattes si anormaux",
                ],
            },
            "humain": {
                "label": "Blessure humaine",
                "keywords": ["main", "bras", "jambe", "pied", "doigt", "peau", "plaie", "brulure", "brûlure", "blessure", "sang"],
                "capture_guidance": [
                    "Vue générale de la zone touchée",
                    "Gros plan net de la blessure ou brûlure",
                    "Photo latérale montrant le gonflement, la profondeur ou l'étendue",
                ],
            },
        }

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", (text or "").lower())
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def _prepare_image(self, image_data: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.thumbnail((256, 256))
        return image

    def _extract_visual_features(self, image: Image.Image) -> Dict[str, float]:
        rgb = np.asarray(image, dtype=np.float32)
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)

        hue = hsv[..., 0] * (360.0 / 255.0)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        gray = rgb.mean(axis=2) / 255.0

        green_mask = (hue >= 45) & (hue <= 150) & (saturation > 0.20) & (value > 0.16)
        yellow_mask = (hue >= 32) & (hue <= 72) & (saturation > 0.22) & (value > 0.28)
        orange_mask = (hue >= 8) & (hue <= 30) & (saturation > 0.32) & (value > 0.20)
        brown_mask = (hue >= 10) & (hue <= 40) & (saturation > 0.25) & (value >= 0.10) & (value <= 0.65)
        dark_mask = value < 0.22
        white_mask = (saturation < 0.15) & (value > 0.72)
        red_mask = ((hue <= 12) | (hue >= 340)) & (saturation > 0.32) & (value > 0.20)
        blue_purple_mask = (hue >= 210) & (hue <= 300) & (saturation > 0.18) & (value > 0.12)
        skin_mask = (
            (((hue >= 0) & (hue <= 35)) | ((hue >= 340) & (hue <= 360)))
            & (saturation >= 0.12)
            & (saturation <= 0.65)
            & (value >= 0.28)
            & (value <= 0.96)
        )

        horizontal_diff = np.abs(np.diff(gray, axis=1))
        vertical_diff = np.abs(np.diff(gray, axis=0))
        edge_density = float(
            (
                (horizontal_diff > 0.16).mean()
                + (vertical_diff > 0.16).mean()
            ) / 2.0
        )

        lesion_mask = brown_mask | (dark_mask & (saturation > 0.18))

        return {
            "green_ratio": float(green_mask.mean()),
            "yellow_ratio": float(yellow_mask.mean()),
            "orange_ratio": float(orange_mask.mean()),
            "brown_ratio": float(brown_mask.mean()),
            "dark_ratio": float(dark_mask.mean()),
            "white_ratio": float(white_mask.mean()),
            "red_ratio": float(red_mask.mean()),
            "blue_purple_ratio": float(blue_purple_mask.mean()),
            "skin_ratio": float(skin_mask.mean()),
            "lesion_ratio": float(lesion_mask.mean()),
            "brightness": float(value.mean()),
            "saturation": float(saturation.mean()),
            "texture": float(gray.std()),
            "edge_density": edge_density,
        }

    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        hits = sum(1 for keyword in keywords if self._normalize_text(keyword) in text)
        return min(0.18, hits * 0.06)

    def _build_visual_observations(self, features: Dict[str, float]) -> List[str]:
        observations: List[str] = []

        if features["green_ratio"] > 0.18:
            observations.append("La photo contient une forte présence de feuillage vert exploitable pour le diagnostic.")
        if features["yellow_ratio"] > 0.14:
            observations.append("Présence notable de jaunissement sur la zone analysée.")
        if features["orange_ratio"] > 0.04:
            observations.append("Des zones orange/brun clair ressemblant à des pustules ou taches sont visibles.")
        if features["brown_ratio"] > 0.10 or features["lesion_ratio"] > 0.12:
            observations.append("Des nécroses ou taches sombres/brunes sont détectées.")
        if features["white_ratio"] > 0.08:
            observations.append("Des zones pâles ou blanchâtres sont visibles sur l'image.")
        if features["red_ratio"] > 0.06:
            observations.append("Des zones rouges ou inflammatoires ressortent sur la photo.")
        if features.get("blue_purple_ratio", 0.0) > 0.05:
            observations.append("Des zones bleu-violet évoquant un hématome ou une contusion sont visibles.")
        if features.get("skin_ratio", 0.0) > 0.16:
            observations.append("La photo contient une zone cutanée bien visible, utile pour une analyse de blessure.")
        if features["texture"] < 0.08:
            observations.append("L'image semble peu contrastée; un diagnostic plus précis demanderait une photo plus nette.")

        return observations[:4]

    def _detect_subject_profile(self, normalized_text: str, normalized_category: str) -> Optional[Dict[str, Any]]:
        for key, profile in self.subject_profiles.items():
            if any(self._normalize_text(keyword) in normalized_text for keyword in profile["keywords"]):
                return {"key": key, **profile}

        if normalized_category == "elevage":
            return {"key": "betail", **self.subject_profiles["betail"]}
        if normalized_category == "sos_accident":
            return {"key": "humain", **self.subject_profiles["humain"]}
        if normalized_category == "agriculture":
            return {"key": "mais", **self.subject_profiles["mais"]}
        return None

    def _infer_diagnosis_type(self, disease: str, subject_profile: Optional[Dict[str, Any]]) -> Dict[str, str]:
        disease_key = self._normalize_text(disease)
        profile_key = subject_profile.get("key") if subject_profile else None

        if profile_key == "humain" or any(token in disease_key for token in ["brulure", "brûlure", "plaie", "contusion", "hematome", "hématome"]):
            return {
                "diagnosis_type": "human_first_aid",
                "diagnosis_type_label": "Blessure humaine / premiers secours",
            }
        if profile_key in {"betail", "petit_ruminant", "volaille", "lapin"} or any(token in disease_key for token in ["animale", "volaille", "sabot", "veterinaire", "vétérinaire", "lapin"]):
            return {
                "diagnosis_type": "animal_health_injury",
                "diagnosis_type_label": "Maladie ou blessure animale",
            }
        return {
            "diagnosis_type": "plant_disease_stress",
            "diagnosis_type_label": "Maladie ou stress de plante",
        }

    def _build_critical_alert(self, disease: str, urgency: str, diagnosis_type: str) -> Dict[str, Any]:
        disease_key = self._normalize_text(disease)
        critical_alert = None
        emergency_actions: List[str] = []
        severity_label = "Surveillance"

        if diagnosis_type == "human_first_aid" and any(token in disease_key for token in ["saignement", "plaie ouverte", "brulure", "brûlure"]):
            critical_alert = "Cas potentiellement urgent: appliquez immédiatement les premiers gestes et cherchez une aide médicale si l'état est grave ou s'aggrave."
            emergency_actions = [
                "Comprimer ou refroidir la zone selon le type de blessure.",
                "Utiliser uniquement de l'eau propre et un tissu propre si disponible.",
                "Contacter les secours ou un centre de santé si la douleur, le saignement ou l'étendue est importante.",
            ]
            severity_label = "Alerte immédiate"
        elif diagnosis_type == "animal_health_injury" and any(token in disease_key for token in ["plaie", "sabot", "oculaire", "variole", "fievre", "fièvre"]):
            critical_alert = "Suspicion de cas animal sérieux: isolez l'animal et faites confirmer rapidement par un agent d'élevage ou vétérinaire."
            emergency_actions = [
                "Isoler l'animal ou la volaille atteinte si possible.",
                "Limiter les déplacements et surveiller écoulement, boiterie, abattement ou difficulté respiratoire.",
                "Désinfecter le matériel et éviter le contact rapproché avec le reste du troupeau.",
            ]
            severity_label = "A surveiller d'urgence"
        elif diagnosis_type == "plant_disease_stress" and urgency == "high":
            critical_alert = "Risque d'aggravation rapide de la culture: isolez ou retirez les parties très atteintes et confirmez vite sur le terrain."
            emergency_actions = [
                "Éviter la propagation par contact ou arrosage sur le feuillage.",
                "Retirer les parties ou plants très atteints si la maladie se diffuse rapidement.",
                "Prendre une seconde photo de confirmation des feuilles, tiges ou fruits.",
            ]
            severity_label = "Intervention rapide"
        elif urgency == "high":
            severity_label = "Intervention rapide"
        elif urgency == "medium":
            severity_label = "À surveiller"

        return {
            "severity_label": severity_label,
            "critical_alert": critical_alert,
            "emergency_actions": emergency_actions,
        }

    def _build_local_context(self, disease: str, diagnosis_type: str, subject_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        disease_key = self._normalize_text(disease)
        subject_label = subject_profile["label"] if subject_profile else None

        if diagnosis_type == "human_first_aid":
            return {
                "local_context_note": "Conseils adaptés à un contexte terrain Burkina: priorité à l'eau propre, au tissu propre, à la protection contre la poussière et à l'accès rapide au centre de santé le plus proche.",
                "local_examples": [
                    "Au champ ou au marché, une coupure sale doit être rincée vite avec eau propre avant pansement propre.",
                    "Pour une brûlure à l'huile ou au feu de cuisson, refroidir à l'eau propre sans appliquer de poudre ou de pâte agressive.",
                    "Si la blessure continue à saigner ou empêche de bouger, cherchez une prise en charge médicale sans attendre.",
                ],
            }

        if diagnosis_type == "animal_health_injury":
            examples = [
                "En élevage villageois, isolez vite l'animal atteint pour limiter la contagion ou l'aggravation.",
                "Nettoyez la zone touchée avec un antiseptique adapté si disponible et gardez l'abri plus sec et propre.",
                "Si l'animal ne mange plus, boite fort ou présente écoulement/fièvre, faites intervenir rapidement un agent d'élevage ou vétérinaire.",
            ]
            if subject_label == "Volaille":
                examples[1] = "Pour la volaille, séparez immédiatement les sujets atteints et désinfectez mangeoires, abreuvoirs et poulailler."
            return {
                "local_context_note": "Conseils orientés élevage Burkina: gestion du troupeau ou du lot, isolement rapide, hygiène de l'abri et recours à l'agent d'élevage local.",
                "local_examples": examples,
            }

        plant_examples = [
            "En saison humide, évitez l'arrosage direct du feuillage déjà taché et retirez vite les parties très atteintes.",
            "Au champ, observez aussi les plants voisins pour voir si le problème se diffuse sur la ligne ou la parcelle.",
            "Si possible, combinez traitement local, aération et rotation culturale au prochain cycle.",
        ]
        if "mais" in disease_key:
            plant_examples[0] = "Sur maïs, comparez plusieurs feuilles du bas et du haut pour distinguer carence et maladie foliaire." 
        elif "manioc" in disease_key:
            plant_examples[0] = "Sur manioc, évitez d'utiliser comme boutures des tiges venant de plants suspects ou déjà desséchés."
        elif "tomate" in disease_key:
            plant_examples[0] = "Sur tomate, aérez davantage les plants et évitez de mouiller les feuilles en fin de journée."

        return {
            "local_context_note": "Conseils formulés pour des pratiques agricoles courantes au Burkina: observation de parcelle, retrait ciblé, hygiène culturale et confirmation par agent agricole si propagation rapide.",
            "local_examples": plant_examples,
        }

    def _analyze_human_condition(self, normalized_text: str, features: Dict[str, float], observations: List[str],
                                 subject_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        bleeding_score = 0.28 + min(0.34, features["red_ratio"] * 2.3) + min(0.16, features["edge_density"] * 0.8)
        bleeding_score += self._keyword_score(normalized_text, self.diseases_database["human_plaie_hemorragique"]["confidence_keywords"])

        burn_score = 0.24 + min(0.22, features["red_ratio"] * 1.5) + min(0.16, features["white_ratio"] * 1.2) + min(0.08, features["orange_ratio"] * 0.9)
        burn_score += self._keyword_score(normalized_text, self.diseases_database["human_brule"]["confidence_keywords"])

        infected_score = 0.22 + min(0.22, features["red_ratio"] * 1.2) + min(0.22, features["yellow_ratio"] * 1.0) + min(0.10, features["brown_ratio"] * 0.8)
        infected_score += self._keyword_score(normalized_text, self.diseases_database["human_infection_plaie"]["confidence_keywords"])

        bruise_score = 0.20 + min(0.28, features.get("blue_purple_ratio", 0.0) * 2.1) + min(0.10, features["dark_ratio"] * 0.6)
        bruise_score += self._keyword_score(normalized_text, self.diseases_database["human_contusion"]["confidence_keywords"])

        if bleeding_score >= max(burn_score, infected_score, bruise_score) and bleeding_score >= 0.42:
            disease = self.diseases_database["human_plaie_hemorragique"]
            return self._base_result(
                disease=disease["name"],
                confidence=bleeding_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les indices visuels et le contexte évoquent une plaie ouverte ou un saignement nécessitant des gestes de premiers secours rapides.",
                recommendations="Si le saignement est abondant, non contrôlé, ou si la plaie est profonde, contactez immédiatement les secours ou un soignant.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if burn_score >= max(infected_score, bruise_score) and burn_score >= 0.40:
            disease = self.diseases_database["human_brule"]
            return self._base_result(
                disease=disease["name"],
                confidence=burn_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="L'image évoque davantage une brûlure superficielle ou intermédiaire qu'une simple coupure mécanique.",
                recommendations="Ajoutez une photo montrant l'étendue complète et précisez la cause (eau chaude, feu, produit chimique, huile, métal chaud).",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if infected_score >= bruise_score and infected_score >= 0.38:
            disease = self.diseases_database["human_infection_plaie"]
            return self._base_result(
                disease=disease["name"],
                confidence=infected_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="La rougeur, les tons jaunâtres ou brunâtres et le contexte texte suggèrent une plaie inflammatoire ou déjà infectée.",
                recommendations="Consultez rapidement s'il y a fièvre, pus, douleur croissante ou extension de la rougeur.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if bruise_score >= 0.34:
            disease = self.diseases_database["human_contusion"]
            return self._base_result(
                disease=disease["name"],
                confidence=bruise_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="L'aspect bleu-violet ou sombre détecté ressemble à une contusion ou un hématome après choc.",
                recommendations="Si la zone ne peut plus bouger normalement, si la douleur est très forte ou si un os semble touché, cherchez une prise en charge médicale.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        return self._base_result(
            disease="Blessure humaine non caractérisée",
            confidence=0.36,
            symptoms=["Lésion visible mais difficile à classer précisément sur photo seule"],
            treatment="Nettoyez à l'eau propre si possible, protégez la zone et consultez si douleur, saignement ou brûlure importante.",
            prevention="Utiliser une protection adaptée et éviter de manipuler la plaie avec des mains sales.",
            urgency="medium",
            analysis="La photo montre bien une atteinte cutanée humaine, mais le type exact de blessure ne peut pas être confirmé avec assez de certitude.",
            recommendations="Ajoutez une autre vue plus nette et précisez la cause, l'heure de l'accident et l'intensité de la douleur.",
            requires_expert=True,
            features=features,
            observations=observations,
            subject_profile=subject_profile,
        )

    def _analyze_animal_condition(self, normalized_text: str, features: Dict[str, float], observations: List[str],
                                  subject_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        subject_key = subject_profile["key"] if subject_profile else None
        wound_score = 0.26 + min(0.24, features["red_ratio"] * 1.8) + min(0.18, features["edge_density"] * 0.7)
        wound_score += self._keyword_score(normalized_text, self.diseases_database["animal_plaie"]["confidence_keywords"])

        skin_infection_score = 0.22 + min(0.18, features["brown_ratio"] * 1.0) + min(0.14, features["yellow_ratio"] * 0.9) + min(0.12, features["texture"] * 0.8)
        skin_infection_score += self._keyword_score(normalized_text, self.diseases_database["animal_infection_cutanee"]["confidence_keywords"])

        ocular_score = 0.21 + min(0.16, features["white_ratio"] * 1.1) + min(0.16, features["red_ratio"] * 1.0)
        ocular_score += self._keyword_score(normalized_text, self.diseases_database["animal_oculaire"]["confidence_keywords"])
        if subject_profile and subject_profile.get("key") == "volaille":
            ocular_score += 0.06

        fever_score = 0.34 + self._keyword_score(normalized_text, self.diseases_database["animal_fievre"]["confidence_keywords"])
        foot_score = 0.24 + min(0.18, features["red_ratio"] * 1.1) + min(0.20, features["brown_ratio"] * 1.1) + min(0.12, features["edge_density"] * 0.6)
        foot_score += self._keyword_score(normalized_text, self.diseases_database["animal_pied_lesion"]["confidence_keywords"])
        if subject_key in {"betail", "petit_ruminant"}:
            foot_score += 0.06

        pox_score = 0.22 + min(0.20, features["brown_ratio"] * 1.0) + min(0.16, features["red_ratio"] * 0.9) + min(0.10, features["white_ratio"] * 0.8)
        pox_score += self._keyword_score(normalized_text, self.diseases_database["volaille_variole"]["confidence_keywords"])
        if subject_key == "volaille":
            pox_score += 0.10

        if foot_score >= max(wound_score, skin_infection_score, ocular_score, fever_score, pox_score) and foot_score >= 0.40:
            disease = self.diseases_database["animal_pied_lesion"]
            return self._base_result(
                disease=disease["name"],
                confidence=foot_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Le contexte de boiterie ou de pied/sabot associé aux indices visuels évoque une lésion localisée du membre.",
                recommendations="Ajoutez une photo de dessous et de profil du pied ou sabot si l'animal accepte de se laisser observer.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if pox_score >= max(wound_score, skin_infection_score, ocular_score, fever_score) and pox_score >= 0.39:
            disease = self.diseases_database["volaille_variole"]
            return self._base_result(
                disease=disease["name"],
                confidence=pox_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Chez la volaille, les croûtes visibles au niveau de la tête ou de la crête évoquent une suspicion de variole aviaire.",
                recommendations="Ajoutez une photo de face et précisez si plusieurs volailles présentent des croûtes similaires.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if wound_score >= max(skin_infection_score, ocular_score, fever_score, foot_score, pox_score) and wound_score >= 0.40:
            disease = self.diseases_database["animal_plaie"]
            return self._base_result(
                disease=disease["name"],
                confidence=wound_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="La photo montre des signes compatibles avec une blessure, une plaie ouverte ou un traumatisme local chez l'animal.",
                recommendations="Vérifiez s'il y a boiterie, écoulement, odeur inhabituelle ou infestation par les mouches.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if skin_infection_score >= max(ocular_score, fever_score, foot_score, pox_score) and skin_infection_score >= 0.38:
            disease = self.diseases_database["animal_infection_cutanee"]
            return self._base_result(
                disease=disease["name"],
                confidence=skin_infection_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les croûtes, taches brunâtres ou altérations de texture orientent vers une atteinte cutanée infectieuse ou parasitaire.",
                recommendations="Ajoutez une photo du contour de la lésion et précisez s'il y a démangeaison, chute de poils ou mauvaise odeur.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if ocular_score >= max(fever_score, foot_score, pox_score) and ocular_score >= 0.35:
            disease = self.diseases_database["animal_oculaire"]
            return self._base_result(
                disease=disease["name"],
                confidence=ocular_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Le contexte et les indices visuels évoquent une atteinte des yeux, des narines ou de la tête chez l'animal.",
                recommendations="Prenez une vue de face en lumière naturelle et signalez toux, abattement ou difficulté respiratoire si présents.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        disease = self.diseases_database["animal_fievre"]
        return self._base_result(
            disease=disease["name"],
            confidence=fever_score,
            symptoms=disease["symptoms"],
            treatment=disease["treatment"],
            prevention=disease["prevention"],
            urgency=disease["urgency"],
            analysis="Le contexte évoque un problème sanitaire animal plus général, sans signe photo assez spécifique pour une confirmation visuelle forte.",
            recommendations="Ajoutez la température, l'appétit, la durée d'évolution et une vue plus nette de la zone anormale.",
            requires_expert=True,
            features=features,
            observations=observations,
            subject_profile=subject_profile,
        )

    def _analyze_plant_condition(self, normalized_text: str, features: Dict[str, float], observations: List[str],
                                 subject_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        subject_key = subject_profile["key"] if subject_profile else None
        nitrogen_score = 0.28 + min(0.28, features["yellow_ratio"] * 1.4) + min(0.10, features["green_ratio"] * 0.4)
        nitrogen_score += self._keyword_score(normalized_text, self.diseases_database["mais_taches_jaunes"]["confidence_keywords"])

        rust_score = 0.24 + min(0.30, features["orange_ratio"] * 2.2) + min(0.12, features["lesion_ratio"] * 0.9)
        rust_score += self._keyword_score(normalized_text, self.diseases_database["mais_rouille"]["confidence_keywords"])

        blight_score = 0.24 + min(0.26, features["brown_ratio"] * 1.5) + min(0.14, features["dark_ratio"] * 0.8)
        blight_score += self._keyword_score(normalized_text, self.diseases_database["tomate_mildiou"]["confidence_keywords"])

        mosaic_score = 0.22 + min(0.24, features["yellow_ratio"] * 0.9) + min(0.12, features["texture"] * 0.8)
        mosaic_score += 0.08 if "manioc" in normalized_text else 0.0

        drought_score = 0.20 + min(0.26, features["brown_ratio"] * 1.1) + min(0.16, (1 - features["brightness"]) * 0.5)
        drought_score += 0.06 if any(token in normalized_text for token in ["seche", "fletri", "manque d'eau", "sol"]) else 0.0

        charbon_score = 0.18 + min(0.34, features["dark_ratio"] * 1.9) + min(0.12, features["texture"] * 0.7)
        charbon_score += self._keyword_score(normalized_text, self.diseases_database["sorgho_charbon"]["confidence_keywords"])

        maize_blight_score = 0.22 + min(0.24, features["brown_ratio"] * 1.3) + min(0.18, features["lesion_ratio"] * 1.1) + min(0.10, features["edge_density"] * 0.5)
        maize_blight_score += self._keyword_score(normalized_text, self.diseases_database["mais_helminthosporiose"]["confidence_keywords"])

        cassava_blight_score = 0.21 + min(0.22, features["brown_ratio"] * 1.0) + min(0.18, features["yellow_ratio"] * 0.8) + min(0.12, features["dark_ratio"] * 0.6)
        cassava_blight_score += self._keyword_score(normalized_text, self.diseases_database["manioc_bacteriose"]["confidence_keywords"])

        oignon_score = 0.20 + min(0.22, features["yellow_ratio"] * 0.8) + min(0.18, features["brown_ratio"] * 1.1)
        arachide_score = 0.19 + min(0.22, features["brown_ratio"] * 1.0) + min(0.10, features["orange_ratio"] * 0.9)

        if subject_key == "tomate":
            blight_score += 0.08
        if subject_key == "manioc":
            mosaic_score += 0.10
            cassava_blight_score += 0.08
        if subject_key == "sorgho":
            charbon_score += 0.10
        if subject_key == "mais":
            rust_score += 0.06
            nitrogen_score += 0.05
            maize_blight_score += 0.09
        if subject_key == "oignon":
            oignon_score += 0.10
        if subject_key == "arachide":
            arachide_score += 0.10

        if maize_blight_score >= max(nitrogen_score, rust_score, blight_score, mosaic_score, drought_score, charbon_score, oignon_score, arachide_score, cassava_blight_score) and maize_blight_score >= 0.40:
            disease = self.diseases_database["mais_helminthosporiose"]
            return self._base_result(
                disease=disease["name"],
                confidence=maize_blight_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les taches brunes et allongées sur le maïs orientent plutôt vers une brûlure foliaire fongique qu'une simple rouille ou carence.",
                recommendations="Ajoutez une photo d'une feuille entière, idéalement avec plusieurs lésions alignées, pour confirmer l'aspect allongé des taches.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if cassava_blight_score >= max(nitrogen_score, rust_score, blight_score, mosaic_score, drought_score, charbon_score, oignon_score, arachide_score) and cassava_blight_score >= 0.39:
            disease = self.diseases_database["manioc_bacteriose"]
            return self._base_result(
                disease=disease["name"],
                confidence=cassava_blight_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Le brunissement et le dessèchement visibles sur le manioc évoquent davantage une brûlure bactérienne qu'une simple mosaïque.",
                recommendations="Prenez aussi une photo des tiges et précisez si le dessèchement progresse rapidement sur plusieurs plants.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if charbon_score >= max(nitrogen_score, rust_score, blight_score, mosaic_score, drought_score, oignon_score, arachide_score, maize_blight_score, cassava_blight_score) and charbon_score >= 0.42:
            disease = self.diseases_database["sorgho_charbon"]
            return self._base_result(
                disease=disease["name"],
                confidence=charbon_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les zones noires et poudreuses détectées sont compatibles avec une suspicion de charbon du sorgho.",
                recommendations="Prenez une photo de l'épi entier pour confirmer si les grains sont remplacés par une masse noire poudreuse.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if oignon_score >= max(nitrogen_score, rust_score, blight_score, mosaic_score, drought_score, arachide_score, maize_blight_score, cassava_blight_score) and oignon_score >= 0.38:
            return self._base_result(
                disease="Suspicion de mildiou ou brûlure foliaire de l'oignon",
                confidence=oignon_score,
                symptoms=[
                    "Jaunissement ou brunissement des feuilles",
                    "Affaiblissement progressif du feuillage",
                ],
                treatment="Réduire l'humidité sur le feuillage, améliorer l'aération et retirer les feuilles très atteintes.",
                prevention="Espacer les plants, éviter l'arrosage tardif sur les feuilles et pratiquer la rotation.",
                urgency="medium",
                analysis="Le profil visuel des feuilles d'oignon suggère une atteinte foliaire plutôt qu'une simple carence.",
                recommendations="Ajoutez une photo du collet et précisez si l'humidité est élevée ou si l'attaque se diffuse rapidement.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if arachide_score >= max(nitrogen_score, rust_score, blight_score, mosaic_score, drought_score, maize_blight_score, cassava_blight_score) and arachide_score >= 0.36:
            return self._base_result(
                disease="Suspicion de cercosporiose ou taches foliaires de l'arachide",
                confidence=arachide_score,
                symptoms=[
                    "Petites taches brunes sur folioles",
                    "Dégradation progressive du feuillage",
                ],
                treatment="Retirer les feuilles très atteintes et appliquer un traitement fongique adapté si disponible localement.",
                prevention="Rotation culturale, semences saines et limitation de l'humidité stagnante.",
                urgency="medium",
                analysis="Les taches détectées sur le feuillage d'arachide évoquent une maladie foliaire fongique courante.",
                recommendations="Prenez un gros plan des deux faces de la foliole pour mieux distinguer tache foliaire et carence.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if rust_score >= max(nitrogen_score, blight_score, mosaic_score, drought_score, maize_blight_score, cassava_blight_score) and rust_score >= 0.45:
            disease = self.diseases_database["mais_rouille"]
            return self._base_result(
                disease=disease["name"],
                confidence=rust_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les zones orange/brunes détectées sur le feuillage sont compatibles avec une suspicion de rouille.",
                recommendations="Photographiez aussi le revers des feuilles pour confirmer la présence de pustules poudreuses.",
                requires_expert=rust_score < 0.62,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if blight_score >= max(nitrogen_score, mosaic_score, drought_score, maize_blight_score, cassava_blight_score) and blight_score >= 0.43:
            disease = self.diseases_database["tomate_mildiou"]
            return self._base_result(
                disease=disease["name"],
                confidence=blight_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Les taches sombres et brunes détectées évoquent une atteinte foliaire de type mildiou ou nécrose avancée.",
                recommendations="Isolez les plants atteints et prenez une seconde photo des tiges et fruits pour confirmer l'extension.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if mosaic_score >= max(nitrogen_score, drought_score, cassava_blight_score, maize_blight_score) and mosaic_score >= 0.40:
            return self._base_result(
                disease="Suspicion de mosaïque foliaire",
                confidence=mosaic_score,
                symptoms=[
                    "Alternance de zones vertes et jaunâtres",
                    "Aspect irrégulier ou marbré du feuillage",
                ],
                treatment="Éliminer les plants très atteints et éviter toute bouture issue de plants suspects.",
                prevention="Utiliser du matériel végétal sain, contrôler les insectes vecteurs et désinfecter les outils.",
                urgency="high",
                analysis="Le motif visuel mêlant vert et jaune sur le feuillage fait penser à une mosaïque virale ou un stress foliaire sévère.",
                recommendations="Ajoutez une photo d'une feuille entière sur fond neutre pour distinguer mosaïque virale et carence.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if nitrogen_score >= max(drought_score, maize_blight_score, cassava_blight_score) and nitrogen_score >= 0.42:
            disease = self.diseases_database["mais_taches_jaunes"]
            return self._base_result(
                disease=disease["name"],
                confidence=nitrogen_score,
                symptoms=disease["symptoms"],
                treatment=disease["treatment"],
                prevention=disease["prevention"],
                urgency=disease["urgency"],
                analysis="Le jaunissement dominant détecté sur l'image évoque davantage une carence nutritive qu'une brûlure localisée.",
                recommendations="Vérifiez si le jaunissement commence sur les feuilles basses et ajoutez si possible l'âge de la culture.",
                requires_expert=nitrogen_score < 0.58,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        if drought_score >= 0.36:
            return self._base_result(
                disease="Stress hydrique ou problème de sol",
                confidence=drought_score,
                symptoms=[
                    "Brunissement ou dessèchement des bords",
                    "Perte de vigueur visuelle",
                ],
                treatment="Contrôler l'humidité du sol, améliorer le paillage et ajuster l'irrigation selon le stade de la culture.",
                prevention="Maintenir une humidité plus stable, apporter de la matière organique et éviter le compactage du sol.",
                urgency="medium",
                analysis="L'image suggère surtout un stress abiotiques: manque d'eau, chaleur ou déséquilibre du sol.",
                recommendations="Prenez aussi une photo du sol au pied de la plante et précisez la fréquence d'arrosage.",
                requires_expert=False,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )

        return self._base_result(
            disease="Indéterminé",
            confidence=0.34,
            symptoms=["Informations visuelles insuffisantes pour identifier précisément la maladie"],
            treatment="Un expert peut confirmer le diagnostic si vous ajoutez une photo plus proche et une description plus détaillée.",
            prevention="Prendre plusieurs photos: vue générale, gros plan des zones atteintes et face inférieure des feuilles.",
            urgency="medium",
            analysis="Une photo a bien été analysée, mais les indices visuels ne permettent pas encore de conclure avec un niveau de confiance suffisant.",
            recommendations="Ajoutez le nom de la culture ou de l'animal, l'ancienneté du problème et une photo plus nette pour améliorer la détection.",
            requires_expert=True,
            features=features,
            observations=observations,
            subject_profile=subject_profile,
        )

    def _base_result(self, *, disease: str, confidence: float, symptoms: List[str], treatment: str,
                     prevention: str, urgency: str, analysis: str, recommendations: str,
                     requires_expert: bool, features: Dict[str, float], observations: List[str],
                     subject_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        diagnosis_meta = self._infer_diagnosis_type(disease, subject_profile)
        alert_meta = self._build_critical_alert(disease, urgency, diagnosis_meta["diagnosis_type"])
        local_context = self._build_local_context(disease, diagnosis_meta["diagnosis_type"], subject_profile)
        return {
            "disease_detected": disease,
            "confidence": round(max(0.0, min(confidence, 0.92)), 2),
            "symptoms": symptoms,
            "treatment": treatment,
            "prevention": prevention,
            "urgency": urgency,
            "analysis": analysis,
            "recommendations": recommendations,
            "requires_expert": requires_expert,
            "analysis_mode": "hybrid_local_image_text",
            "visual_observations": observations,
            "visual_features": {key: round(value, 3) for key, value in features.items()},
            "detected_subject": subject_profile["label"] if subject_profile else None,
            "capture_guidance": subject_profile["capture_guidance"] if subject_profile else None,
            "diagnosis_type": diagnosis_meta["diagnosis_type"],
            "diagnosis_type_label": diagnosis_meta["diagnosis_type_label"],
            "severity_label": alert_meta["severity_label"],
            "critical_alert": alert_meta["critical_alert"],
            "emergency_actions": alert_meta["emergency_actions"],
            "local_context_note": local_context["local_context_note"],
            "local_examples": local_context["local_examples"],
        }

    def analyze_image_simple(self, image_data: bytes, text_description: str = "", category: Optional[str] = None) -> dict:
        """
        Analyse locale hybride: indices visuels extraits de l'image + contexte texte.
        """
        try:
            normalized_text = self._normalize_text(text_description)
            normalized_category = self._normalize_text(category or "")
            image = self._prepare_image(image_data)
            features = self._extract_visual_features(image)
            observations = self._build_visual_observations(features)
            subject_profile = self._detect_subject_profile(normalized_text, normalized_category)

            is_human_context = normalized_category == "sos_accident" or any(
                token in normalized_text for token in ["blessure", "plaie", "coupure", "brulure", "brûlure", "sang", "main", "bras", "jambe", "pied", "doigt"]
            )
            is_animal_context = not is_human_context and (normalized_category == "elevage" or any(
                token in normalized_text for token in ["animal", "betail", "vache", "mouton", "chevre", "volaille", "peau"]
            ))
            is_plant_context = not is_human_context and not is_animal_context and (
                normalized_category in {"agriculture", ""}
                or any(token in normalized_text for token in ["feuille", "plante", "culture", "mais", "tomate", "manioc", "sorgho", "tache"])
                or features["green_ratio"] > 0.12
            )

            if is_human_context:
                return self._analyze_human_condition(
                    normalized_text,
                    features,
                    observations,
                    subject_profile,
                )

            if is_animal_context:
                return self._analyze_animal_condition(
                    normalized_text,
                    features,
                    observations,
                    subject_profile,
                )

            if is_plant_context:
                return self._analyze_plant_condition(
                    normalized_text,
                    features,
                    observations,
                    subject_profile,
                )

            return self._base_result(
                disease="Indéterminé",
                confidence=0.34,
                symptoms=["Informations visuelles insuffisantes pour identifier précisément la maladie"],
                treatment="Un expert peut confirmer le diagnostic si vous ajoutez une photo plus proche et une description plus détaillée.",
                prevention="Prendre plusieurs photos: vue générale, gros plan des zones atteintes et face inférieure des feuilles.",
                urgency="medium",
                analysis="Une photo a bien été analysée, mais les indices visuels ne permettent pas encore de conclure avec un niveau de confiance suffisant.",
                recommendations="Ajoutez le nom de la culture ou de l'animal, l'ancienneté du problème et une photo plus nette pour améliorer la détection.",
                requires_expert=True,
                features=features,
                observations=observations,
                subject_profile=subject_profile,
            )
        except Exception as e:
            return {
                "disease_detected": "Erreur d'analyse",
                "confidence": 0.0,
                "symptoms": ["Erreur technique"],
                "treatment": "Veuillez décrire le problème par texte.",
                "prevention": "Réessayez avec une photo plus claire.",
                "urgency": "low",
                "analysis": f"Erreur technique lors de l'analyse: {str(e)}",
                "recommendations": "Veuillez réessayer ou décrire le problème par texte.",
                "requires_expert": True,
                "analysis_mode": "hybrid_local_image_text",
            }

    def analyze_images(self, images_data: List[bytes], text_description: str = "", category: Optional[str] = None) -> Dict[str, Any]:
        valid_images = [image for image in images_data if image][:3]
        if not valid_images:
            raise ValueError("Aucune photo exploitable fournie")

        results = [
            self.analyze_image_simple(image, text_description, category)
            for image in valid_images
        ]

        ranked = sorted(
            enumerate(results, start=1),
            key=lambda item: (
                item[1].get("confidence", 0.0),
                0 if item[1].get("requires_expert") else 1,
                len(item[1].get("visual_observations") or []),
            ),
            reverse=True,
        )
        best_view_index, best_result = ranked[0]
        aggregated = dict(best_result)

        same_disease_count = sum(
            1
            for result in results
            if result.get("disease_detected") == best_result.get("disease_detected")
        )
        if same_disease_count >= 2:
            aggregated["confidence"] = round(
                min(0.95, aggregated.get("confidence", 0.0) + 0.05),
                2,
            )

        aggregated["photo_count"] = len(valid_images)
        aggregated["best_view_index"] = best_view_index
        aggregated["analyzed_views"] = [
            {
                "view_index": index,
                "disease_detected": result.get("disease_detected"),
                "confidence": result.get("confidence"),
                "analysis": result.get("analysis"),
            }
            for index, result in enumerate(results, start=1)
        ]
        if aggregated.get("photo_count", 1) > 1:
            aggregated["analysis"] = (
                f"{aggregated.get('analysis', '')} Analyse consolidée sur {aggregated['photo_count']} vues; "
                f"la vue {best_view_index} apporte les indices les plus nets."
            ).strip()

        return aggregated


def _collect_photo_payloads(primary_photo: Optional[str], photo_list: Optional[List[str]]) -> List[str]:
    payloads: List[str] = []
    for payload in ([primary_photo] if primary_photo else []) + (photo_list or []):
        if not payload:
            continue
        if payload not in payloads:
            payloads.append(payload)
    return payloads[:3]


def _decode_photo_payload(photo_string: str) -> bytes:
    if "," in photo_string:
        photo_string = photo_string.split(",", 1)[1]
    return base64.b64decode(photo_string)


def _load_json_list(raw: Optional[str]) -> List[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _build_upload_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"http://localhost:8000/{path}"


def _store_photo_payloads(owner_id: int, photo_data_list: List[bytes], prefix: str = "photo") -> List[str]:
    stored_paths: List[str] = []
    timestamp = int(datetime.utcnow().timestamp())
    for index, photo_data in enumerate(photo_data_list, start=1):
        filename = f"{owner_id}_{timestamp}_{prefix}_{index}.jpg"
        path = f"uploads/{filename}"
        with open(path, "wb") as f:
            f.write(photo_data)
        stored_paths.append(path)
    return stored_paths


def _serialize_photo_history_record(record: PhotoAnalysisHistoryDB) -> Dict[str, Any]:
    photo_paths = _load_json_list(record.photo_paths_json)
    photo_labels = _load_json_list(record.photo_labels_json)
    photos = [
        {
            "url": _build_upload_url(path),
            "label": photo_labels[index] if index < len(photo_labels) else f"Vue {index + 1}",
        }
        for index, path in enumerate(photo_paths)
        if path
    ]

    analysis: Dict[str, Any] = {}
    if record.analysis_json:
        try:
            parsed = json.loads(record.analysis_json)
            if isinstance(parsed, dict):
                analysis = parsed
        except Exception:
            analysis = {}

    return {
        "id": record.client_record_id or str(record.id),
        "server_id": record.id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "category": record.category or "agriculture",
        "prompt": record.prompt or "",
        "analysis": analysis,
        "photos": photos,
        "source_ticket_id": record.source_ticket_id,
    }


class GPTVisionEngine:
    """Analyse d'images via GPT-4 Vision API (OpenAI)
    
    Remplace le Computer Vision local pour bénéficier des capacités
    d'analyse avancée de ChatGPT avec sa propre connaissance.
    """
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.model = "gpt-4o"
    
    def analyze_images(self, images_data: List[bytes], text_description: str = "", category: Optional[str] = None) -> Dict[str, Any]:
        """Analyser les images via GPT-4 Vision API"""
        valid_images = [image for image in images_data if image][:3]
        if not valid_images:
            raise ValueError("Aucune photo exploitable fournie")
        
        try:
            # Convertir les images en base64
            images_base64 = [base64.b64encode(img).decode('utf-8') for img in valid_images]
            
            # Créer le prompt contextuel
            context_prompt = ""
            if category == "agriculture":
                context_prompt = """TÂCHE: Analyser une photo agricole pour identifier les maladies des cultures.

INSTRUCTIONS IMPORTANTES:
- Analysez la photo avec ATTENTION aux détails
- Identifiez chaque culture visible
- Détectez TOUTE maladie, ravageur ou problème visible
- Si AUCUN problème: écrivez "Aucune maladie détectée"
- Donnez une confiance entre 0.0 (aucune certitude) et 1.0 (certitude totale)
- IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre

FORMAT JSON REQUIS (copiez-collez et remplissez):
{
    "disease_detected": "Maladie identifiée ou 'Aucune'",
    "confidence": 0.85,
    "symptoms": ["Symptôme 1", "Symptôme 2"],
    "treatment": "Action recommandée",
    "urgency": "low|medium|high",
    "prevents": "Prévention",
    "visual_observations": ["Détail observé"],
    "analysis": "Explication détaillée en français"
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

FORMAT JSON REQUIS (copiez-collez et remplissez):
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
            
            # Construire le message pour GPT-4 Vision
            content = [
                {
                    "type": "text",
                    "text": context_prompt + (f"\n\nContext supplémentaire: {text_description}" if text_description else "")
                }
            ]
            
            # Ajouter les images
            for idx, img_b64 in enumerate(images_base64):
                print(f"📸 Image {idx + 1}: {len(img_b64)} bytes ({len(img_b64)/1024:.1f}KB) - {category} mode")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
            
            print(f"📤 Envoi à GPT-4o: {len(content)} éléments (1 texte + {len(images_base64)} images)")
            
            # Appeler GPT-4o avec vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=1024,
                temperature=0.7
            )
            
            print(f"✅ Réponse GPT-4o reçue")
            
            # Parser la réponse
            response_text = response.choices[0].message.content
            print(f"📝 Réponse brute GPT-4o: {response_text[:300]}...")
            
            # Extraire le JSON de la réponse
            analysis_json = None
            try:
                # Chercher le JSON dans la réponse
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    analysis_json = json.loads(json_str)
                    print(f"✅ JSON parsé avec succès: {analysis_json.get('disease_detected', 'N/A')}")
                else:
                    print(f"⚠️ Pas de JSON trouvé dans la réponse")
                    
            except json.JSONDecodeError as je:
                print(f"⚠️ JSON parsing error: {je}")
                # Essayer de nettoyer et re-parser
                try:
                    # Remplacer characteres problématiques
                    cleaned = response_text.replace('\n', ' ').replace('  ', ' ')
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        analysis_json = json.loads(json_match.group())
                        print(f"✅ JSON parsé après nettoyage: {analysis_json.get('disease_detected', 'N/A')}")
                except:
                    print(f"❌ Impossible de parser JSON après nettoyage")
            
            # Si toujours pas de JSON valide, créer response par défaut
            if not analysis_json:
                # Essayer d'extraire quelques mots-clés utiles
                disease_keywords = ['maladie', 'malade', 'blessure', 'infection', 'aucun', 'aucune', 'normal', 'sain']
                detected = 'Non identifiée'
                for keyword in disease_keywords:
                    if keyword in response_text.lower():
                        detected = 'Détecté' if keyword not in ['aucun', 'aucune', 'normal', 'sain'] else 'Aucune maladie'
                        break
                
                print(f"⚠️ Utilisant response par défaut avec keyword matching")
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
            
            return analysis_json
            
        except Exception as e:
            print(f"❌ Erreur GPT-4o Vision: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "disease_detected": "Erreur analyse",
                "confidence": 0,
                "analysis": f"Erreur lors de l'analyse: {str(e)}",
                "urgency": "medium",
                "requires_expert": True,
                "photo_count": len(valid_images),
                "error": str(e)
            }


cv_engine = GeminiVisionEngine(GEMINI_API_KEY) if GEMINI_API_KEY else (GPTVisionEngine(openai_client) if openai_client else LocalComputerVision())

# ==========================================
# MODULE IA TEXTE (NLP Local) - RESTAURÉ
# ==========================================

class AITriageEngine:
    def __init__(self):
        self.urgency_keywords = {
            "high": ["urgence", "urgent", "grave", "danger", "sang", "brûlure", "piraté", 
                    "volé", "mort", "mourir", "pourrir", "invasion", "attaque"],
            "medium": ["problème", "aide", "rapidement", "besoin", "important", "malade"],
            "low": ["conseil", "information", "question", "quand", "comment", "préventif"]
        }
        
        self.category_keywords = {
            "agriculture": ["maïs", "sorgho", "mil", "culture", "plante", "champ", "récolte", 
                          "irrigation", "tomate", "oignon", "arachide", "coton",
                          "manioc", "riz", "feuille", "insecte", "parasite", "engrais"],
            # Catégorie élevage : animaux, bétail, poules…
            "elevage": ["bétail", "vache", "boeuf", "chèvre", "mouton", "poules", "volaille",
                        "lapin", "lapereau", "clapier",
                        "agneau", "veau", "animal", "troupeau", "abri", "vermifuge", "parasites"],
            # Catégorie SOS Accident / premiers soins : on garde le domaine health pour le RAG
            "sos_accident": ["blessure", "coupure", "accident", "saigne", "sang", "brûlure",
                              "tomber", "chute", "fracture", "douleur", "secours"],
            "cybersecurity": ["arnaque", "pirate", "mobile money", "code", "mot de passe", 
                            "orange money", "sms suspect", "compte", "fraude", "virus"]
        }
    
    def classify(self, text: str):
        text_lower = text.lower()
        
        # Catégorie
        category_scores = {}
        for cat, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            category_scores[cat] = score
        
        category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else "agriculture"
        confidence = category_scores[category] / (len(self.category_keywords[category]) + 1)
        
        # Urgence
        urgency = "low"
        for level, keywords in self.urgency_keywords.items():
            if any(kw in text_lower for kw in keywords):
                urgency = level
                break
        
        keywords = [word for word in text_lower.split() if len(word) > 3][:5]
        
        return {
            "category": category,
            "urgency": urgency,
            "confidence": float(confidence),
            "keywords": keywords
        }

ai_engine = AITriageEngine()

FOCUS_SUBJECTS: Dict[str, List[Dict[str, Any]]] = {
    "agriculture": [
        {"label": "Maïs", "aliases": ["maïs", "mais"]},
        {"label": "Tomate", "aliases": ["tomate", "tomates"]},
        {"label": "Manioc", "aliases": ["manioc", "bouture de manioc"]},
        {"label": "Sorgho", "aliases": ["sorgho", "sorghos"]},
        {"label": "Oignon", "aliases": ["oignon", "oignons"]},
        {"label": "Arachide", "aliases": ["arachide", "arachides", "cacahuète", "cacahuete"]},
        {"label": "Riz", "aliases": ["riz", "rizière", "riziere"]},
        {"label": "Mil", "aliases": ["mil"]},
        {"label": "Coton", "aliases": ["coton"]},
    ],
    "elevage": [
        {"label": "Lapin", "aliases": ["lapin", "lapins", "lapereau", "lapereaux", "clapier"]},
        {"label": "Volaille", "aliases": ["volaille", "poule", "poules", "poulet", "poulets", "coq", "canard"]},
        {"label": "Vache", "aliases": ["vache", "vaches", "boeuf", "boeufs", "bovin", "bovins", "veau", "veaux"]},
        {"label": "Chèvre", "aliases": ["chèvre", "chevre", "chèvres", "chevres"]},
        {"label": "Mouton", "aliases": ["mouton", "moutons", "brebis", "agneau", "agneaux"]},
        {"label": "Porc", "aliases": ["porc", "porcs", "cochon", "cochons"]},
    ],
    "sos_accident": [
        {"label": "Main", "aliases": ["main", "mains", "doigt", "doigts", "paume"]},
        {"label": "Bras", "aliases": ["bras", "coude"]},
        {"label": "Jambe", "aliases": ["jambe", "jambes", "genou", "cuisse"]},
        {"label": "Pied", "aliases": ["pied", "pieds", "orteil", "orteils"]},
        {"label": "Œil", "aliases": ["oeil", "œil", "yeux"]},
        {"label": "Peau", "aliases": ["peau", "visage", "tête", "tete"]},
    ],
    "cybersecurity": [
        {"label": "Orange Money", "aliases": ["orange money", "orangemoney"]},
        {"label": "Moov Money", "aliases": ["moov money", "moovmoney"]},
        {"label": "Mobile Money", "aliases": ["mobile money", "wallet"]},
        {"label": "WhatsApp", "aliases": ["whatsapp", "whats app"]},
        {"label": "Facebook", "aliases": ["facebook", "fb"]},
        {"label": "Téléphone", "aliases": ["telephone", "téléphone", "portable", "smartphone"]},
        {"label": "Carte SIM", "aliases": ["carte sim", "sim"]},
    ],
}

FOCUS_ISSUES: Dict[str, List[Dict[str, Any]]] = {
    "agriculture": [
        {"label": "Taches jaunes", "aliases": ["taches jaunes", "tache jaune", "jaunissement", "jaune"]},
        {"label": "Rouille", "aliases": ["rouille", "pustule orange", "pustules orange"]},
        {"label": "Mildiou", "aliases": ["mildiou", "pourriture", "feuilles noires"]},
        {"label": "Mosaïque", "aliases": ["mosaïque", "mosaique", "feuilles déformées", "feuilles deformees"]},
        {"label": "Bactériose", "aliases": ["bacteriose", "bactériose", "brûlure bactérienne", "brulure bacterienne"]},
    ],
    "elevage": [
        {"label": "Plaie", "aliases": ["plaie", "blessure", "saigne", "coupure"]},
        {"label": "Boiterie", "aliases": ["boite", "boiterie", "patte", "sabot", "pied"]},
        {"label": "Fièvre", "aliases": ["fièvre", "fievre", "chaud", "abattu"]},
        {"label": "Infection cutanée", "aliases": ["croûte", "croute", "peau", "gale", "plaque"]},
        {"label": "Atteinte oculaire", "aliases": ["oeil", "œil", "écoulement", "ecoulement", "narine"]},
        {"label": "Diarrhée", "aliases": ["diarrhee", "diarrhée", "selles liquides"]},
    ],
    "sos_accident": [
        {"label": "Plaie ouverte", "aliases": ["plaie", "coupure", "saigne", "saignement"]},
        {"label": "Brûlure", "aliases": ["brûlure", "brulure", "huile chaude", "feu"]},
        {"label": "Fracture", "aliases": ["fracture", "cassé", "casse", "déboîté", "deboite"]},
        {"label": "Infection", "aliases": ["pus", "infecté", "infecte", "gonflé", "gonfle"]},
        {"label": "Contusion", "aliases": ["bleu", "choc", "contusion", "hématome", "hematome"]},
    ],
    "cybersecurity": [
        {"label": "Arnaque", "aliases": ["arnaque", "escroquerie", "fraude"]},
        {"label": "Pirâtage", "aliases": ["piraté", "pirate", "hacking", "compte volé", "compte vole"]},
        {"label": "Code OTP", "aliases": ["otp", "code secret", "code de validation"]},
        {"label": "SIM swap", "aliases": ["sim swap", "carte sim", "reseau perdu", "réseau perdu"]},
        {"label": "Virus", "aliases": ["virus", "lien suspect", "application suspecte"]},
    ],
}


# ==========================================
# APPLICATION FASTAPI
# ==========================================

app = FastAPI(
    title="SONGRA API - IA Locale",
    version="5.0",
    description="Plateforme d'assistance avec IA locale pour l'analyse de photos"
)

# CORS - AUTORISER TOUT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def create_access_token(user: User) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": str(user.id),
        "phone_number": user.phone_number,
        "type": "user",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=JWT_EXPIRE_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    return token


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    token = _extract_bearer_token(authorization)

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    if payload.get("type") != "user":
        raise HTTPException(status_code=401, detail="Invalid token scope")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


def get_current_expert(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> Expert:
    token = _extract_bearer_token(authorization)

    if not token.startswith("token_"):
        raise HTTPException(status_code=401, detail="Invalid expert token")

    parts = token.split("_", 2)
    if len(parts) < 3 or not parts[1].isdigit():
        raise HTTPException(status_code=401, detail="Invalid expert token")

    expert = db.query(Expert).filter(Expert.id == int(parts[1])).first()
    if not expert:
        raise HTTPException(status_code=401, detail="Expert not found")

    return expert


def serialize_user(user: User) -> Dict[str, Any]:
    return {
        "id": user.id,
        "phone_number": user.phone_number,
        "name": user.name,
        "location": user.location,
        "is_premium": user.is_premium,
        "messages_used": user.messages_used,
        "messages_limit": user.messages_limit if user.is_premium else 1,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _validate_user_credentials(phone_number: str, password: str) -> None:
    if not (phone_number or "").strip():
        raise HTTPException(status_code=422, detail="Phone number is required")
    if len((password or "").strip()) < 4:
        raise HTTPException(status_code=422, detail="Password must contain at least 4 characters")


# ==========================================
# BASE DE CONNAISSANCE (RAG SIMPLE)
# ==========================================

# INITIALISATION AU DÉMARRAGE
# ==========================================

@app.on_event("startup")
async def startup_seed_data():
    """Initialiser les données minimales au démarrage, y compris sous Gunicorn."""
    try:
        db = SessionLocal()
        existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
        if not existing:
            expert = Expert(
                email="test@resolvehub.bf",
                password_hash=hash_password("test123"),
                full_name="Expert Test IA",
                specialization="agriculture",
                is_active=True
            )
            db.add(expert)
            db.commit()
            print("✓ Expert test créé: test@resolvehub.bf / test123")

        try:
            load_knowledge_from_json(db)
            total_items = db.query(KnowledgeItem).count()
            print(f"✓ Base de connaissances chargée ({total_items} fiches)")
        except Exception as e_load:
            print(f"⚠️ Erreur chargement base de connaissances: {e_load}")
        finally:
            db.close()
    except Exception as e:
        print(f"[WARN] Erreur initialisation startup: {e}")


def load_knowledge_from_json(db: Session, file_path: str = "knowledge_base.json") -> None:
    """Charger une base de connaissances simple à partir d'un fichier JSON.

    Le fichier doit contenir une liste d'objets de la forme :
    {
        "domain": "agriculture" | "elevage" | "cybersecurity" | "health",
        "title": "Titre court compréhensible par un agriculteur",
        "question": "Formulation typique de la question",
        "answer": "Réponse détaillée, validée par les experts locaux",
        "tags": ["mais", "taches jaunes", "engrais"],
        "language": "fr",
        "source": "ONG locale",  # optionnel
    }
    """
    # Toujours se baser sur le dossier du fichier main.py pour trouver le JSON,
    # afin que ça fonctionne même si le serveur est lancé depuis la racine.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(file_path):
        file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de charger {file_path}: {e}")
        return

    if not isinstance(raw, list):
        print("⚠️ knowledge_base.json doit contenir une liste d'entrées")
        return

    for item in raw:
        try:
            _upsert_knowledge_item(
                db,
                domain=item.get("domain", "agriculture"),
                title=item.get("title"),
                question=item.get("question"),
                answer=item.get("answer"),
                tags=item.get("tags") or [],
                language=item.get("language", "fr"),
                source=item.get("source"),
                media=item.get("media"),
            )
        except Exception as e:
            print(f"⚠️ Erreur lors de l'import d'une entrée de connaissance: {e}")

    db.commit()


def _upsert_knowledge_item(
    db: Session,
    domain: str,
    title: Optional[str],
    answer: Optional[str],
    question: Optional[str] = None,
    tags: Optional[List[str]] = None,
    language: str = "fr",
    source: Optional[str] = None,
    media: Optional[Any] = None,
) -> None:
    """Mettre à jour la KB en considérant le titre comme clé canonique.

    Cela répare les bases déjà polluées où une même fiche a pu être enregistrée
    sous un mauvais domaine ou en doublon. Au prochain démarrage/import, la
    fiche est réalignée sur le domaine déclaré dans le JSON.
    """
    if not title or not answer:
        return

    normalized_domain = (domain or "agriculture").strip().lower()
    serialized_tags = json.dumps(tags or [], ensure_ascii=False)
    serialized_media = json.dumps(media, ensure_ascii=False) if media is not None else None

    existing_items = (
        db.query(KnowledgeItem)
        .filter(KnowledgeItem.title == title)
        .order_by(KnowledgeItem.id.asc())
        .all()
    )

    primary = next(
        (item for item in existing_items if (item.domain or "").strip().lower() == normalized_domain),
        None,
    )
    if primary is None and existing_items:
        primary = existing_items[0]

    if primary is None:
        db.add(
            KnowledgeItem(
                domain=normalized_domain,
                title=title,
                question=question,
                answer=answer,
                tags=serialized_tags,
                language=language,
                source=source,
                media=serialized_media,
            )
        )
        return

    primary.domain = normalized_domain
    primary.question = question
    primary.answer = answer
    primary.tags = serialized_tags
    primary.language = language
    primary.source = source
    primary.media = serialized_media

    for duplicate in existing_items:
        if duplicate.id != primary.id:
            db.delete(duplicate)


def _normalize_token(token: str) -> str:
    """Normaliser grossièrement un mot français pour améliorer le matching.

    - Passe en minuscules
    - Supprime les accents (maïs -> mais)
    - Supprime un "s" final (pluriel simple : jaunes -> jaune)
    """
    if not token:
        return ""

    # Minuscules + suppression des accents
    token = unicodedata.normalize("NFD", token.lower())
    token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")

    # Supprimer la ponctuation pour éviter les "jaune," vs "jaune"
    token = "".join(ch for ch in token if ch.isalnum())

    # Pluriel très simple : retirer un "s" final
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]

    return token


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    raw_tokens = text.split()
    tokens: List[str] = []
    for w in raw_tokens:
        norm = _normalize_token(w)
        if len(norm) > 2:
            tokens.append(norm)
    return tokens


def _normalize_free_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(normalized.split())


def _build_focus_terms(label: Optional[str], aliases: Optional[List[str]] = None) -> List[str]:
    values = [label or "", *(aliases or [])]
    terms: List[str] = []
    for value in values:
        normalized_value = _normalize_free_text(value)
        if normalized_value and normalized_value not in terms:
            terms.append(normalized_value)
        for token in _tokenize(value):
            if token not in terms:
                terms.append(token)
    return terms


def _find_best_focus_match(category: str, text: str, focus_map: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    candidates = focus_map.get(category, [])
    if not candidates:
        return None

    normalized_text = _normalize_free_text(text)
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    best_alias_length = 0

    for candidate in candidates:
        aliases = candidate.get("aliases", [])
        matched_aliases = []
        for alias in aliases:
            normalized_alias = _normalize_free_text(alias)
            if normalized_alias and normalized_alias in normalized_text:
                matched_aliases.append(alias)

        score = len(matched_aliases)
        alias_length = max((len(_normalize_free_text(alias)) for alias in matched_aliases), default=0)
        if score > best_score or (score == best_score and alias_length > best_alias_length):
            best = candidate
            best_score = score
            best_alias_length = alias_length

    if not best:
        return None

    return {
        "label": best["label"],
        "aliases": best.get("aliases", []),
        "terms": _build_focus_terms(best["label"], best.get("aliases", [])),
    }


def extract_focus_context(
    category: str,
    text: str,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    subject = _find_best_focus_match(category, text, FOCUS_SUBJECTS)
    issue = _find_best_focus_match(category, text, FOCUS_ISSUES)

    if photo_analysis:
        detected_subject = photo_analysis.get("detected_subject")
        if detected_subject:
            subject = {
                "label": str(detected_subject),
                "aliases": [str(detected_subject)],
                "terms": _build_focus_terms(str(detected_subject)),
            }

        disease_detected = photo_analysis.get("disease_detected")
        if disease_detected:
            issue = {
                "label": str(disease_detected),
                "aliases": [str(disease_detected)],
                "terms": _build_focus_terms(str(disease_detected)),
            }

    return {
        "subject": subject,
        "issue": issue,
    }


def _focus_label(focus_context: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    focus_entry = (focus_context or {}).get(key)
    if isinstance(focus_entry, dict):
        label = focus_entry.get("label")
        return str(label) if label else None
    return None


def _build_precise_no_match_answer(domain: str, focus_context: Optional[Dict[str, Any]] = None) -> str:
    """Réponse quand aucune fiche RAG n'est trouvée - Songra/Yingr-AI reste humble."""
    focus_subject = _focus_label(focus_context, "subject")
    focus_issue = _focus_label(focus_context, "issue")

    focus_parts = [label for label in [focus_subject, focus_issue] if label]
    if focus_parts:
        target = " / ".join(focus_parts)
        return (
            f"Moi, Songra (assistant de Yingr-AI), je n'ai pas trouvé une fiche assez précise pour ton cas : {target}. "
            "Je préfère rester honnête plutôt que de généraliser vers un autre animal, une autre culture ou un autre problème. \n"
            "Ce que je te conseille : "
            "Ajoute si possible un symptôme clé, une photo plus nette de la situation, "
            "ou rapproche-toi d'un expert local (agent agricole, vétérinaire, service de santé du Burkina Faso) pour une vérification sur place."
        )

    domain_label = {
        "agriculture": "agriculture",
        "elevage": "élevage",
        "health": "premiers secours",
        "cybersecurity": "cybersécurité",
    }.get(domain, domain)
    return (
        f"Je suis Songra, l'assistant IA de Yingr-AI dédié au Burkina Faso. "
        f"Je n'ai pas trouvé de fiche suffisamment précise dans le domaine {domain_label}. "
        "Accumuler des connaissances locales du Burkina Faso est mon mission, "
        "mais pour cette question spécifique, je dois rester prudent. \n"
        "Ajoute des détails concrets sur ton problème (quelle culture, quel animal, quelle région au BF?) "
        "pour que je puisse mieux t'aider. "
        "Ou contacte un expert local du domaine pour une vérification fiable."
    )


def retrieve_knowledge(
    db: Session,
    domain: str,
    query: str,
    limit: int = 5,
    expand_scope: bool = True,
    focus_subject: Optional[Dict[str, Any]] = None,
    focus_issue: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Récupération améliorée basée sur le recouvrement de mots-clés pondéré.

    - Les correspondances dans le titre et les tags comptent plus que celles
      présentes uniquement dans la réponse longue.
        - La recherche est limitée au domaine demandé. Aucun mélange automatique
            entre agriculture et élevage n'est autorisé.
        - Si ``expand_scope`` est activé et qu'aucune fiche n'est trouvée dans le
            domaine demandé, on fait un second passage sur toutes les fiches pour
            éviter de rater une correspondance évidente.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    focus_subject_terms = focus_subject.get("terms", []) if focus_subject else []
    focus_issue_terms = focus_issue.get("terms", []) if focus_issue else []

    def score_items(items: List[KnowledgeItem]) -> List[Dict[str, Any]]:
        scored_local: List[Dict[str, Any]] = []
        for it in items:
            # Séparer les zones de texte pour mieux pondérer
            title_tokens = set(_tokenize(it.title or ""))
            question_tokens = set(_tokenize(it.question or ""))
            answer_tokens = set(_tokenize(it.answer or ""))

            tags_list: List[str] = []
            if it.tags:
                try:
                    tags_list = json.loads(it.tags)
                except Exception:
                    tags_list = []
            tags_tokens = set(_tokenize(" ".join(tags_list))) if tags_list else set()

            overlap_title = len(query_tokens & title_tokens)
            overlap_question = len(query_tokens & question_tokens)
            overlap_answer = len(query_tokens & answer_tokens)
            overlap_tags = len(query_tokens & tags_tokens)

            # Pondération simple : titre > tags > question > réponse
            score = (
                overlap_title * 3.0
                + overlap_tags * 2.5
                + overlap_question * 2.0
                + overlap_answer * 1.0
            )

            combined_text = _normalize_free_text(
                f"{it.title}\n{it.question or ''}\n{it.answer}\n{' '.join(tags_list)}"
            )
            subject_match = any(term in combined_text for term in focus_subject_terms) if focus_subject_terms else False
            issue_match = any(term in combined_text for term in focus_issue_terms) if focus_issue_terms else False

            if focus_subject_terms and subject_match:
                score += 8.0
            if focus_issue_terms and issue_match:
                score += 4.5

            if score <= 0:
                continue

            scored_local.append({
                "item": it,
                "score": score,
                "subject_match": subject_match,
                "issue_match": issue_match,
            })

        return scored_local

    # 1) Fiches strictement dans le domaine demandé
    primary_query = db.query(KnowledgeItem).filter(KnowledgeItem.domain == domain)

    primary_items = primary_query.all()
    scored = score_items(primary_items)

    # 2) Fallback global optionnel : si rien trouvé, on regarde toutes les fiches
    if not scored and expand_scope:
        all_items = db.query(KnowledgeItem).all()
        scored = score_items(all_items)

    # 3) Dernier recours : recherche par sous-chaîne, soit dans le domaine
    # strict, soit sur toute la base si l'élargissement est autorisé.
    if not scored:
        fallback_items = primary_items if not expand_scope else db.query(KnowledgeItem).all()

        def normalize_text(text: str) -> str:
            if not text:
                return ""
            tokens = _tokenize(text)
            return " ".join(tokens)

        norm_query_parts = list(query_tokens)
        for it in fallback_items:
            big_text = f"{it.title}\n{it.question or ''}\n{it.answer}\n"
            if it.tags:
                try:
                    tags_list = json.loads(it.tags)
                except Exception:
                    tags_list = []
                big_text += " ".join(tags_list)
            norm_text = normalize_text(big_text)
            if any(part in norm_text for part in norm_query_parts):
                scored.append({"item": it, "score": 1.0})

    if not expand_scope:
        scored = [
            entry
            for entry in scored
            if (entry["item"].domain or "").strip().lower() == domain.strip().lower()
        ]

    if focus_subject_terms and any(entry.get("subject_match") for entry in scored):
        scored = [entry for entry in scored if entry.get("subject_match")]

    if focus_issue_terms and any(entry.get("issue_match") for entry in scored):
        scored = [entry for entry in scored if entry.get("issue_match")]

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_items = [s["item"] for s in scored[:limit]]

    results: List[Dict[str, Any]] = []
    for it in top_items:
        media_data: Optional[Any] = None
        if it.media:
            try:
                media_data = json.loads(it.media)
            except Exception:
                media_data = None

        results.append({
            "id": it.id,
            "domain": it.domain,
            "title": it.title,
            "question": it.question,
            "answer": it.answer,
            "tags": _load_json_list(it.tags),
            "language": it.language,
            "source": it.source,
            "media": media_data,
        })

    return results


def generate_llm_answer_with_general_knowledge(
    question: str,
    language: str,
    domain: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Assistant rural utilisant la connaissance générale de ChatGPT.
    
    Quand il n'y a pas de fiche dans la base de connaissances locale,
    Songra utilise ses connaissances générales pour aider la communauté.
    Elle reste un assistant rural accessible et pratique.
    
    📷 Si une photo a été analysée, son diagnostic guide la réponse.
    """
    if not openai_client or not OPENAI_API_KEY:
        return None

    conversation_text = ""
    if conversation_context:
        serialized_turns = []
        for turn in conversation_context[-6:]:
            role = "Utilisateur" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if content:
                serialized_turns.append(f"{role}: {content}")
        if serialized_turns:
            conversation_text = "\n\nContexte de la conversation :\n" + "\n".join(serialized_turns) + "\n"

    # Construire le diagnostic photo s'il existe
    # ⚠️ SI UNE PHOTO, C'EST LE CŒUR DE LA RÉPONSE
    photo_diagnosis_section = ""
    has_photo_diagnosis = False
    if photo_analysis and (photo_analysis.get("disease_detected") or photo_analysis.get("observations")):
        has_photo_diagnosis = True
        diagnosis_parts = []
        
        if photo_analysis.get("disease_detected"):
            diagnosis_parts.append(f"Problème : {photo_analysis.get('disease_detected')}")
        
        if photo_analysis.get("detected_subject"):
            diagnosis_parts.append(f"Sujet : {photo_analysis.get('detected_subject')}")
        
        if photo_analysis.get("observations"):
            diagnosis_parts.append(f"Observations : {photo_analysis.get('observations')}")
        
        if photo_analysis.get("urgency"):
            urgency_fr = {
                "immediate": "IMMÉDIATE 🚨",
                "high": "ÉLEVÉE ⚠️",
                "medium": "MOYENNE",
                "low": "BASSE"
            }.get(photo_analysis.get("urgency"), photo_analysis.get("urgency"))
            diagnosis_parts.append(f"Urgence : {urgency_fr}")
        
        if diagnosis_parts:
            photo_diagnosis_section = (
                "📸 DIAGNOSTIC DE TA PHOTO :\n"
                + " | ".join(diagnosis_parts) + 
                "\n"
            )

    domain_description = {
        "agriculture": "l'agriculture et les cultures",
        "elevage": "l'élevage et l'élevage du bétail",
        "health": "les premiers secours et la sécurité sanitaire",
        "cybersecurity": "la cybersécurité et la sécurité en ligne",
    }.get(domain, domain)

    system_prompt = (
        "Tu es Songra, un assistant rural qui aide les communautés à résoudre leurs difficultés quotidiennes. \n"
        "Tu travailles avec Yingr-AI, une intelligence artificielle pour soutenir les populations. \n"
        f"Ta spécialité actuelle : {domain_description}. \n\n"
        "Tu es pratique, accessible et toujours prêt à aider. \n"
        "Tes réponses doivent être : \n"
        "- SIMPLES et directes (compréhensible par tout le monde) \n"
        "- PRATIQUES avec des conseils qu'on peut appliquer tout de suite \n"
        "- HONNÊTE sur ce que tu sais et ce que tu ne sais pas \n"
        "- ENCOURAGEANTE : tu crois que la communauté peut réussir \n\n"
        "Tu n'as pas de fiche spécialisée exacte pour cette question, donc tu utilises tes connaissances générales. \n"
        "Mais tu donnes toujours des conseils pratiques et adaptés aux situations réelles. \n"
        "Tu n'inventes jamais, tu dis toujours si tu n'es pas sûr. \n"
    )

    # Construire le prompt selon qu'on a une photo ou pas
    if has_photo_diagnosis:
        # PHOTO ANALYSÉE = point de départ obligatoire
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n\n"
            f"🎯 TU DOIS COMMENCER PAR :\n"
            f"{photo_diagnosis_section}\n"
            "Tâche uniquement :\n"
            "1️⃣ Commence par : 'D'après l'analyse de ta photo :'\n"
            "2️⃣ Décris ce problème (2-3 phrases claires)\n"
            "3️⃣ Actions concrètes & pratiques (numérotées)\n"
            "4️⃣ Dis si expert est VRAIMENT nécessaire\n"
            "5️⃣ Max 10-15 phrases. Langage SIMPLE.\n"
            "6️⃣ Ne dévie JAMAIS du diagnostic photo.\n\n"
            f"{conversation_text}\n"
        )
    else:
        # Sans photo = question textuelle normale
        user_prompt = (
            f"Langue : {language or 'fr'}. Domaine : {domain}.\n"
            f"Question : {question}\n"
            f"{conversation_text}\n"
            "Tâche : Aide cette personne de manière pratique et simple. \n"
            "- Explique ce que tu comprends du problème (2-3 phrases). \n"
            "- Donne des conseils concrets qu'on peut faire tout de suite (numérotés). \n"
            "- Dis si tu penses qu'il faut l'aide d'un expert et pourquoi. \n"
            "- Sois honnête si tu n'es pas totalement sûr. \n"
            "- Utilise un langage simple et pratique. \n"
            "- Limite à 10-15 phrases maximum. \n\n"
            "Réponds TOUJOURS de manière pratique pour aider la communauté à résoudre ses difficultés."
        )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,  # Plus naturel et pratique
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("⚠️ Erreur appel OpenAI (generate_llm_answer_with_general_knowledge):", repr(e))
        return None


def resolve_knowledge_answer(
    db: Session,
    domain: str,
    question: str,
    language: str = "fr",
    conversation_context: Optional[List[Dict[str, str]]] = None,
    limit: int = 5,
    focus_context: Optional[Dict[str, Any]] = None,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Répondre via fallback amélioré : RAG strict (moins strict) → Connaissances générales.

    Stratégie RÉVISÉE (sans mélange de catégories) :
    1. RAG strict dans le domaine : chercher PLUS de fiches (moins de seuil)
    2. Si rien : Utiliser les connaissances générales du LLM (sans mélanger les domaines)
    3. Fallback minimal : réponse générique "Je ne sais pas"
    
    ⚠️ SUPPRESSION VOLONTAIRE : La phase "RAG expanded" qui mélangeait les catégories
    a été enlevée. On passe directement à la connaissances générales de Songra.
    
    📷 AMÉLIORATION PHOTO : Le diagnostic photo (si disponible) est maintenant passé
    au LLM pour enrichir la réponse et forcer la précision sur le diagnostic détecté.
    """
    # ÉTAPE 1 : Recherche RAG strict dans le domaine demandé (avec limit augmenté)
    # Augmenter limit pour chercher plus de fiches dans le bon domaine
    rag_items = retrieve_knowledge(
        db,
        domain,
        question,
        limit=limit + 3,  # +3 pour être moins strict sans mélanger
        expand_scope=False,  # IMPORTANT : rester dans le domaine, pas de mélange
        focus_subject=(focus_context or {}).get("subject"),
        focus_issue=(focus_context or {}).get("issue"),
    )
    
    if rag_items:
        # Succès RAG strict : utiliser ces fiches
        llm_answer = generate_llm_answer(
            question=question,
            language=language,
            domain=domain,
            knowledge_items=rag_items,
            conversation_context=conversation_context,
            focus_context=focus_context,
            photo_analysis=photo_analysis,
        )
        return {
            "rag_items": rag_items,
            "llm_answer": llm_answer,
            "rag_fallback_answer": None if llm_answer else rag_items[0].get("answer"),
            "knowledge_mode": "rag_strict",
            "knowledge_fallback_used": False,
        }

    # ÉTAPE 2 : SAUTÉE (ancien RAG expanded qui mélangeait les catégories)
    # On va directement aux connaissances générales de Songra

    # ÉTAPE 3 : Aucune fiche trouvée → Utiliser connaissances générales du LLM
    # C'est Songra qui répond avec sa propre connaissance, sans inventer de catégories
    if openai_client and OPENAI_API_KEY:
        general_answer = generate_llm_answer_with_general_knowledge(
            question=question,
            language=language,
            domain=domain,
            conversation_context=conversation_context,
            photo_analysis=photo_analysis,
        )
        if general_answer:
            return {
                "rag_items": [],
                "llm_answer": general_answer,
                "rag_fallback_answer": None,
                "knowledge_mode": "llm_general_knowledge",  # Songra avec ses connaissances générales
                "knowledge_fallback_used": True,
            }

    # ÉTAPE 4 : Fallback ultime - aucune source d'info disponible
    return {
        "rag_items": [],
        "llm_answer": None,
        "rag_fallback_answer": _build_precise_no_match_answer(domain, focus_context),
        "knowledge_mode": "no_match",
        "knowledge_fallback_used": False,
    }


def generate_llm_answer(
    question: str,
    language: str,
    domain: str,
    knowledge_items: List[Dict[str, Any]],
    conversation_context: Optional[List[Dict[str, str]]] = None,
    focus_context: Optional[Dict[str, Any]] = None,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Songra (Yingr-AI) reformule et raisonne à partir de la base RAG validée localement au Burkina Faso.

    IMPORTANT : Songra NE DOIT PAS inventer de faits en dehors des fiches RAG du Burkina Faso.
    - S'il n'y a pas assez de fiches, Songra le dit clairement.
    - Toutes les réponses doivent respecter le contexte socio-climatique du BF.
    - Songra reste humble et recommande toujours un expert local si doute.
    - SI une photo a été analysée, sa diagnosis DOIT être au cœur de la réponse.
    """
    if not knowledge_items:
        # Pas de base de connaissance pertinente, on ne force pas le modèle
        return None

    focus_subject_label = _focus_label(focus_context, "subject")
    focus_issue_label = _focus_label(focus_context, "issue")

    # Petit fallback local : si le LLM n'est pas disponible, on formate au
    # minimum une réponse structurée à partir de la meilleure fiche.
    def build_structured_from_rag() -> str:
        best = knowledge_items[0]
        titre = best.get("title") or "Conseil local"
        reponse = best.get("answer") or ""
        source = best.get("source") or "fiches locales"

        parts = []
        focus_hint = ""
        if focus_subject_label or focus_issue_label:
            focus_parts = [label for label in [focus_subject_label, focus_issue_label] if label]
            focus_hint = f" de façon précise sur {' / '.join(focus_parts)}"
        parts.append(
            f"1) Ce que Songra (moi!) comprends de ton problème :\n"
            f"Tu signales un souci lié à : {titre}. "
            f"Je vais te partager les conseils validés et déjà accumulés au Burkina Faso{focus_hint}, "
            f"car je suis Songra, l'assistant IA local de Yingr-AI dédié aux communautés du BF."
        )
        parts.append(
            "2) Conseils pratiques pour le contexte du Burkina Faso :\n" + reponse
        )
        parts.append(
            "3) Quand ABSOLUMENT contacter un expert local :\n"
            "Si malgré ces conseils la situation ne s'améliore pas, si le problème s'aggrave, ou si tu as un doute, "
            "rapproche-toi d'un agent agricole, d'un vétérinaire, ou d'un service technique local pour vérifier sur place. "
            "C'est important pour ta sécurité et celle de ta famille/communauté."
            f" (Conseil validé localement - Source : {source})."
        )
        return "\n\n".join(parts)

    # Si pas de client OpenAI (clé manquante ou désactivée), on renvoie tout
    # de suite la version structurée basée uniquement sur le RAG.
    if not openai_client or not OPENAI_API_KEY:
        return build_structured_from_rag()

    context_blocks = []
    for idx, item in enumerate(knowledge_items, start=1):
        context_blocks.append(
            f"FICHE {idx} ({item.get('domain', '')}) - {item.get('title', '')}\n"
            f"Question typique: {item.get('question', '')}\n"
            f"Réponse validée: {item.get('answer', '')}\n"
            f"Mots-clés: {', '.join(item.get('tags', []))}\n"
        )

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "Tu es Songra, l'agent d'assistance IA de Yingr-AI (Yingr Artificial Intelligence). \n"
        "Yingr-AI est une intelligence artificielle LOCALE et SOUVERAINE basée au Burkina Faso. \n"
        "Tu es le lien entre la connaissance validée et les communautés rurales du Burkina Faso. \n\n"
        "IDENTITÉ ET MISSION : \n"
        "- Tu es Songra, dédié à l'assistance des communautés du Burkina Faso \n"
        "- Tu fournis des conseils en agriculture, élevage, et cybersécurité adaptés au contexte burkinabè \n"
        "- Ton objectif : autonomiser les agriculteurs et éleveurs du BF avec des solutions locales \n\n"
        "CONTEXTE BURKINABÈ - TOUJOURS à l'esprit : \n"
        "- Climat Sahélien avec sécheresses périodiques \n"
        "- Ressources naturelles limitées mais exploitables intelligemment \n"
        "- Cultures principales : mil, sorgho, maïs, arachide, coton, oignons \n"
        "- Élevages : bovins, ovins, caprins, volailles adaptés aux conditions locales \n"
        "- Langues locales : Mooré, Dioula, Fulfuldé dominent les zones rurales \n\n"
        "RÈGLES STRICTES : \n"
        "- Tu dois répondre UNIQUEMENT avec les fiches ci-dessous. \n"
        "- Si les fiches ne suffisent pas, dis-le clairement sans généralisations hasardeuses. \n"
        "- Pas de hors-sujet, zéro conseil médical avancé ou dangereux. \n"
        "- Langage TRÈS simple, phrases courtes, concret, sans jargon - adapté aux populations peu alphabétisées. \n"
        "- Réponds en français clair mais intègre des mots en langue locale si c'est plus approprié \n"
        "(ex: 'zaï' pour les trous en agroforesterie, 'daba' pour la houe). \n"
        "- TOUJOURS recommander de vérifier avec un expert local (agent agricole, vétérinaire, service de santé du BF). \n"
        "- Adapte chaque conseil à la climatologie et aux réalités socio-économiques burkinabè. \n"
    )

    focus_instruction = ""
    if focus_subject_label or focus_issue_label:
        focus_parts = [label for label in [focus_subject_label, focus_issue_label] if label]
        focus_instruction = (
            "\nContrainte de précision : reste strictement centré sur cet objet précis"
            f" : {' / '.join(focus_parts)}. "
            "Ne dérive pas vers des conseils généraux d'une autre espèce, d'une autre culture, d'une autre blessure ou d'un autre service. "
            "Si les fiches ne couvrent pas précisément cet objet, dis-le clairement au lieu de généraliser.\n"
        )

    conversation_text = ""
    if conversation_context:
        serialized_turns = []
        for turn in conversation_context[-6:]:
            role = "Utilisateur" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if content:
                serialized_turns.append(f"{role}: {content}")
        if serialized_turns:
            conversation_text = "\n\nContexte de la conversation en cours :\n" + "\n".join(serialized_turns) + "\n"

    # Construire le diagnostic photo détaillé pour enrichir le prompt
    # ⚠️ SI UNE PHOTO EST ANALYSÉE, C'EST LE POINT DE DÉPART OBLIGATOIRE
    photo_diagnosis_section = ""
    has_photo_diagnosis = False
    if photo_analysis and (photo_analysis.get("disease_detected") or photo_analysis.get("observations")):
        has_photo_diagnosis = True
        diagnosis_parts = []
        
        if photo_analysis.get("disease_detected"):
            diagnosis_parts.append(f"Problème détecté : {photo_analysis.get('disease_detected')}")
        
        if photo_analysis.get("detected_subject"):
            diagnosis_parts.append(f"Sujet identifié : {photo_analysis.get('detected_subject')}")
        
        if photo_analysis.get("observations"):
            diagnosis_parts.append(f"Observations : {photo_analysis.get('observations')}")
        
        if photo_analysis.get("urgency"):
            urgency_fr = {
                "immediate": "IMMÉDIATE 🚨",
                "high": "ÉLEVÉE ⚠️",
                "medium": "MOYENNE",
                "low": "BASSE"
            }.get(photo_analysis.get("urgency"), photo_analysis.get("urgency"))
            diagnosis_parts.append(f"Urgence : {urgency_fr}")
        
        if photo_analysis.get("confidence"):
            diagnosis_parts.append(f"Confiance : {photo_analysis.get('confidence')}%")
        
        if diagnosis_parts:
            photo_diagnosis_section = (
                "📸 DIAGNOSTIC PHOTO ANALYSÉE PAR IA :\n"
                + " | ".join(diagnosis_parts) + 
                "\n"
            )

    user_prompt = ""
    
    # SI photo analysée, c'est LA PRIORITÉ - elle guide TOUT
    if has_photo_diagnosis:
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n\n"
            f"🎯 POINT DE DÉPART OBLIGATOIRE - Analyse photo :\n"
            f"{photo_diagnosis_section}\n"
            f"FICHES DE CONNAISSANCE DISPONIBLES :\n{context_text}\n\n"
            "INSTRUCTIONS STRICTES :\n"
            "1️⃣ Commence OBLIGATOIREMENT par : 'D'après l'analyse de ta photo :'\n"
            "2️⃣ Expose d'abord le diagnostic détecté\n"
            "3️⃣ Donne des actions CONCRÈTES basées sur ce diagnostic\n"
            "4️⃣ Dis clairement si expert est nécessaire\n"
            "5️⃣ Max 10-15 phrases. Langage SIMPLE.\n"
            "6️⃣ Ne varie PAS du diagnostic. S'il y a des fiches pertinentes, base-toi dessus.\n"
            "Si tu n'as pas assez d'infos des fiches, dis-le franchement.\n\n"
            f"{focus_instruction}"
            f"{conversation_text}"
        )
    else:
        # Sans photo analysée - CONSULTATION EN LIGNE COMPLÈTE en texte
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n"
            f"Tu es un consultant expert en ligne SONGRA. L'utilisateur a une question spécifique:\n"
            f"Question: {question}\n\n"
            f"FICHES DE CONNAISSANCE DISPONIBLES (base locale BF validée):\n{context_text}\n\n"
            f"{focus_instruction}"
            f"{conversation_text}\n"
            "✅ STRUCTURE OBLIGATOIRE POUR CONSULTATION EN LIGNE:\n\n"
            "1️⃣ **DIAGNOSTIC & ANALYSE** (2-3 phrases max):\n"
            "   → Ce que tu comprends du problème de l'utilisateur\n"
            "   → Pourquoi c'est important d'agir\n\n"
            "2️⃣ **RECOMMANDATIONS PRATIQUES** (5-7 étapes numérotées 1., 2., 3., etc):\n"
            "   → Actions concrètes que l'utilisateur peut faire MAINTENANT\n"
            "   → Étapes réalistes au Burkina Faso (sans équipement avancé)\n"
            "   → Chaque point doit être actionnable immédiatement\n\n"
            "3️⃣ **SOLUTIONS/RESSOURCES LOCALES** (si applicable):\n"
            "   → Produits ou services disponibles au Burkina Faso\n"
            "   → Remèdes locaux efficaces\n"
            "   → Contacts ou où trouver aide (agent agricole, vétérinaire local, etc)\n\n"
            "4️⃣ **QUAND CONSULTER UN EXPERT** (critères très clairs):\n"
            "   → Signes d'urgence à monitorer\n"
            "   → S'il y a [condition X], impossible d'attendre\n"
            "   → Contact: agent agricole/vétérinaire/service de santé local\n\n"
            "5️⃣ **PRÉVENTION** (1-2 phrases):\n"
            "   → Comment éviter ce problème à l'avenir\n"
            "   → Bonnes pratiques pour durabilité\n\n"
            "⚠️ RÈGLES STRICTES:\n"
            "- Langage TRÈS SIMPLE (populations peu alphabétisées Burkina Faso)\n"
            "- Phrases COURTES (max 15-20 mots par phrase)\n"
            "- PAS de jargon technique sans explication\n"
            "- BASÉ 100% sur les fiches fournies\n"
            "- Si fiches insuffisantes: dis clairement 'Je n'ai pas assez d'infos spécialisées, consulte un expert local pour vérification'\n"
            "- TOUJOURS recommander expert si: situation complexe, doute OU danger potentiel\n"
            "- MAX 15-20 phrases TOTAL (sois concis mais complet)\n"
            "- Utilise français clair + mots locaux si approprié (ex: 'zaï', 'daba', 'vétérinaire')"
        )

    try:
        # Utiliser un modèle GPT standard largement disponible
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Log détaillé pour diagnostiquer les problèmes de modèle / de clé
        print("⚠️ Erreur appel OpenAI (generate_llm_answer):", repr(e))
        # En cas d'erreur LLM (quota, réseau, modèle indisponible...), on
        # revient à la réponse structurée 100% basée sur la fiche locale.
        return build_structured_from_rag()

# ==========================================
# ROUTES API (Avec analyse IA restaurée)
# ==========================================

@app.get("/")
async def root():
    return {
        "service": "Songra - Agent Yingr-AI",
        "description": "Assistant IA local et souverain au service des communautés du Burkina Faso",
        "organization": "Yingr-AI (Yingr Artificial Intelligence)",
        "mission": "Autonomiser les agriculteurs et éleveurs du Burkina Faso avec l'IA locale",
        "specializations": ["Agriculture BF", "Élevage BF", "Cybersécurité BF"],
        "version": "5.0",
        "features": [
            "Analyse IA texte (classification)",
            "Analyse IA photo (Computer Vision local)",
            "Base de connaissances RAG validée localement",
            "Raisonnement contextuel au Burkina Faso",
            "Fallback multi-niveaux (RAG strict → RAG élargi → Connaissances générales)"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "service": "Songra/Yingr-AI",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "ia_status": "active",
        "ai_name": "Songra",
        "ai_role": "Assistant IA dédié aux communautés du Burkina Faso"
    }


@app.post("/api/register")
async def register_user(data: UserRegister, db: Session = Depends(get_db)):
    _validate_user_credentials(data.phone_number, data.password)

    existing = db.query(User).filter(User.phone_number == data.phone_number.strip()).first()
    if existing and existing.password_hash:
        raise HTTPException(status_code=409, detail="User already exists")

    if existing:
        user = existing
        user.name = data.name.strip()
        user.location = (data.location or "").strip() or user.location
        user.password_hash = hash_password(data.password)
    else:
        user = User(
            phone_number=data.phone_number.strip(),
            password_hash=hash_password(data.password),
            name=data.name.strip(),
            location=(data.location or "").strip() or None,
            is_premium=False,
            messages_used=0,
            messages_limit=1,
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    return {
        "token": create_access_token(user),
        "user": serialize_user(user),
    }


@app.post("/api/login")
async def login_user(data: UserLogin, db: Session = Depends(get_db)):
    _validate_user_credentials(data.phone_number, data.password)

    user = db.query(User).filter(User.phone_number == data.phone_number.strip()).first()
    if not user or not user.password_hash or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "token": create_access_token(user),
        "user": serialize_user(user),
    }


@app.get("/api/user")
async def get_authenticated_user(current_user: User = Depends(get_current_user)):
    return serialize_user(current_user)


@app.get("/api/dashboard")
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    recent_questions = await get_user_tickets(current_user.phone_number, db)
    recent_questions = recent_questions[:5]

    preferred_domain = next(
        (item.get("category") for item in recent_questions if item.get("category")),
        "agriculture",
    )
    ai_suggestions = retrieve_knowledge(db, preferred_domain, preferred_domain, limit=3)

    community_messages = (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.is_hidden == False)
        .order_by(ChatMessageDB.created_at.desc())
        .limit(6)
        .all()
    )

    return {
        "user": serialize_user(current_user),
        "recent_questions": recent_questions,
        "ai_suggestions": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "answer": item.get("answer"),
                "domain": item.get("domain"),
                "tags": item.get("tags") or [],
            }
            for item in ai_suggestions
        ],
        "community_activity": [
            {
                "id": message.id,
                "sender": message.sender,
                "text": message.text,
                "is_bot": message.is_bot,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in community_messages
        ],
    }


@app.post("/api/scanner/analyze")
async def analyze_scanner_photo(
    data: MobileQuestionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyse UNIQUEMENT une photo du scanner - pas de ticket, pas de chat IA."""
    
    # Collecter les photos
    photo_payloads = _collect_photo_payloads(data.photo_base64, data.photo_base64_list)
    
    if not photo_payloads:
        raise HTTPException(status_code=400, detail="Aucune photo fournie")
    
    try:
        # Analyser les photos avec Gemini
        photo_data_list = [_decode_photo_payload(payload) for payload in photo_payloads]
        photo_analysis = cv_engine.analyze_images(
            photo_data_list, 
            data.content or "", 
            data.category or "agriculture"
        )
        
        return {
            "status": "success",
            "analysis": photo_analysis,
            "category": data.category,
            "model": photo_analysis.get("model", "gemini-2.5-flash")
        }
        
    except Exception as e:
        print(f"[ERROR] Analyse scanner photo: {e}")
        return {
            "status": "error",
            "error": str(e),
            "analysis": {"error": str(e), "requires_expert": True}
        }


@app.get("/api/questions")
async def list_mobile_questions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await get_user_tickets(current_user.phone_number, db)


@app.post("/api/questions")
async def create_mobile_question(
    data: MobileQuestionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    message_data = MessageCreate(
        content=data.content,
        phone_number=current_user.phone_number,
        channel="app",
        category=data.category,
        photo_base64=data.photo_base64,
        photo_base64_list=data.photo_base64_list,
        conversation_context=data.conversation_context,
    )
    return await incoming_sms(message_data, db)


@app.get("/api/questions/{question_id}")
async def get_mobile_question_detail(
    question_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == question_id, Ticket.user_id == current_user.id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Question not found")

    ticket_detail = await get_ticket_detail(question_id, db)
    ai_summary_data = await get_ticket_ai_summary(question_id, db)

    return {
        "question": ticket_detail["ticket"],
        "user": ticket_detail["user"],
        "messages": ticket_detail["messages"],
        "ai_summary": ai_summary_data.get("ai_summary"),
        "rag_items": ai_summary_data.get("rag_items", []),
        "latest_expert_answer": next(
            (message["content"] for message in reversed(ticket_detail["messages"]) if message.get("sender_type") == "expert"),
            None,
        ),
    }

@app.post("/api/auth/login")
async def login(data: ExpertLogin, db: Session = Depends(get_db)):
    expert = db.query(Expert).filter(Expert.email == data.email).first()
    if not expert or not verify_password(data.password, expert.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "token": f"token_{expert.id}_{datetime.utcnow().timestamp()}",
        "expert": {
            "id": expert.id,
            "name": expert.full_name,
            "email": expert.email,
            "specialization": expert.specialization
        }
    }

@app.get("/api/tickets")
async def get_tickets(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Ticket)
    if status:
        query = query.filter(Ticket.status == status)
    
    tickets = query.order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        user = db.query(User).filter(User.id == ticket.user_id).first()
        last_msg = db.query(Message).filter(
            Message.ticket_id == ticket.id
        ).order_by(Message.sent_at.desc()).first()
        
        # Construire l'URL de la photo
        photo_url = None
        if ticket.photo_path:
            photo_url = f"http://localhost:8000/{ticket.photo_path}"
        photo_paths = _load_json_list(ticket.photo_paths_json)
        photo_urls = [_build_upload_url(path) for path in photo_paths if path]
        
        result.append({
            "id": ticket.id,
            "user_phone": user.phone_number if user else "Inconnu",
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else "Aucun message",
            "ai_confidence": ticket.ai_confidence_score,
            "has_photo": ticket.photo_path is not None,
            "photo_url": photo_url,
            "photo_urls": photo_urls,
            "photo_path": ticket.photo_path,
            "has_photo_analysis": ticket.ai_photo_analysis is not None
        })
    
    return result

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_tickets = db.query(Ticket).count()
    open_tickets = db.query(Ticket).filter(Ticket.status == "open").count()
    assigned_tickets = db.query(Ticket).filter(Ticket.status == "assigned").count()
    
    # Tickets résolus aujourd'hui
    today = datetime.utcnow().date()
    resolved_today = db.query(Ticket).filter(
        Ticket.status == "resolved",
        func.date(Ticket.resolved_at) == today
    ).count()
    
    tickets_with_photos = db.query(Ticket).filter(
        Ticket.photo_path.isnot(None)
    ).count()
    
    return {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "assigned_tickets": assigned_tickets,
        "resolved_today": resolved_today,
        "tickets_with_photos": tickets_with_photos
    }

@app.get("/api/tickets/{ticket_id}")
async def get_ticket_detail(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    user = db.query(User).filter(User.id == ticket.user_id).first()
    messages = db.query(Message).filter(
        Message.ticket_id == ticket_id
    ).order_by(Message.sent_at).all()
    
    # Gérer l'analyse photo
    photo_analysis = None
    if ticket.ai_photo_analysis:
        try:
            if isinstance(ticket.ai_photo_analysis, str):
                photo_analysis = json.loads(ticket.ai_photo_analysis)
            else:
                photo_analysis = ticket.ai_photo_analysis
        except:
            photo_analysis = {"error": "Failed to parse analysis"}
    
    # Construire l'URL de la photo
    photo_url = None
    if ticket.photo_path:
        photo_url = f"http://localhost:8000/{ticket.photo_path}"
    photo_paths = _load_json_list(ticket.photo_paths_json)
    photo_urls = [_build_upload_url(path) for path in photo_paths if path]
    
    # Extraire les mots-clés
    keywords = []
    if ticket.ai_extracted_keywords:
        try:
            keywords = json.loads(ticket.ai_extracted_keywords)
        except:
            keywords = []
    
    return {
        "ticket": {
            "id": ticket.id,
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "keywords": keywords,
            "confidence": ticket.ai_confidence_score or 0.5,
            "photo_url": photo_url,
            "photo_urls": photo_urls,
            "photo_path": ticket.photo_path,
            "photo_filename": ticket.photo_path,  # Alias pour le frontend expert
            "photo_analysis": photo_analysis,
            "created_at": ticket.created_at,
            "resolved_at": ticket.resolved_at
        },
        "user": {
            "phone": user.phone_number if user else None,
            "name": user.name if user else None,
            "location": user.location if user else None
        },
        "messages": [{
            "content": msg.content,
            "sender_type": msg.sender_type,
            "sent_at": msg.sent_at
        } for msg in messages]
    }

@app.post("/api/webhooks/incoming-sms")
async def incoming_sms(data: MessageCreate, db: Session = Depends(get_db)):
    # 1. Trouver ou créer l'utilisateur
    user = db.query(User).filter(User.phone_number == data.phone_number).first()
    if not user:
        user = User(phone_number=data.phone_number)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # 2. Analyse IA texte (RESTAURÉ)
    ai_result = ai_engine.classify(data.content)

    # 2.bis Déterminer la catégorie finale et le domaine RAG
    # On combine la catégorie choisie dans l'app (data.category) et
    # la catégorie NLP locale.
    nlp_category = ai_result.get("category", "agriculture")
    chosen_category = data.category or nlp_category
    ai_result["classifier_category"] = nlp_category
    ai_result["category"] = chosen_category

    # Mapping catégorie -> domaine de la base de connaissances
    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"  # les fiches premiers soins sont dans le domaine health
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"
    
    # 3. Analyse photo si présente (RESTAURÉ)
    photo_analysis = None
    photo_path = None
    photo_paths: List[str] = []
    
    photo_payloads = _collect_photo_payloads(data.photo_base64, data.photo_base64_list)
    if photo_payloads:
        try:
            photo_data_list = [_decode_photo_payload(payload) for payload in photo_payloads]
            photo_analysis_result = cv_engine.analyze_images(photo_data_list, data.content, data.category)
            photo_analysis = json.dumps(photo_analysis_result, ensure_ascii=False)
            
            photo_paths = _store_photo_payloads(user.id, photo_data_list, prefix="ticket")
            best_index = max(0, photo_analysis_result.get("best_view_index", 1) - 1)
            photo_path = photo_paths[best_index] if best_index < len(photo_paths) else photo_paths[0]
            
            # Ajuster urgence si maladie grave détectée
            if photo_analysis_result.get("urgency") == "high":
                ai_result["urgency"] = "high"
                
        except Exception as e:
            print(f"Erreur analyse photo: {e}")
            photo_analysis = json.dumps({"error": str(e), "requires_expert": True})
    
    # 3.bis RAG strict d'abord, base élargie seulement si le RAG ne répond pas.
    photo_analysis_payload = None
    if photo_analysis:
        try:
            photo_analysis_payload = json.loads(photo_analysis)
        except Exception:
            photo_analysis_payload = None

    focus_context = extract_focus_context(
        chosen_category,
        data.content,
        photo_analysis_payload,
    )

    knowledge_result = resolve_knowledge_answer(
        db=db,
        domain=kb_domain,
        question=data.content,
        language="fr",
        focus_context=focus_context,
        photo_analysis=photo_analysis_payload,
    )
    rag_items = knowledge_result["rag_items"]
    llm_answer = knowledge_result["llm_answer"]

    # LOG DEBUG : affichage des fiches RAG et de la réponse LLM (si disponible)
    try:
        print("[RAG] Domaine:", kb_domain)
        knowledge_mode = knowledge_result['knowledge_mode']
        fallback_used = knowledge_result['knowledge_fallback_used']
        
        if knowledge_mode == "rag_strict":
            print(f"✓ [RAG-STRICT] {len(rag_items)} fiche(s) trouvée(s) dans le domaine exact")
        elif knowledge_mode == "rag_expanded":
            print(f"⚠ [RAG-EXPANDED] {len(rag_items)} fiche(s) trouvée(s) (recherche élargie dans d'autres domaines)")
        elif knowledge_mode == "llm_general_knowledge":
            print(f"⚡ [LLM-GENERAL] Pas de fiche RAG, utilisation des connaissances générales de Lia")
        elif knowledge_mode == "no_match":
            print(f"✗ [NO-MATCH] Aucune connaissance trouvée, réponse générique")
        
        for idx, item in enumerate(rag_items, start=1):
            print(f"  - FICHE {idx}: {item.get('title')} (domaine: {item.get('domain')})")
        
        if llm_answer:
            if knowledge_mode == "llm_general_knowledge":
                print("[LLM-GENERAL] Réponse générée avec connaissances générales (début):", llm_answer[:250].replace("\n", " "))
            else:
                print("[LLM] Réponse générée à partir de RAG (début):", llm_answer[:250].replace("\n", " "))
        else:
            print("[LLM] Aucune réponse LLM générée")
    except Exception as e_log:
        print(f"[DEBUG] Erreur lors du log RAG/LLM: {e_log}")

    # 4. Créer ticket avec analyse IA
    ticket = Ticket(
        user_id=user.id,
        category=chosen_category,
        urgency=ai_result["urgency"],
        ai_confidence_score=ai_result["confidence"],
        ai_extracted_keywords=json.dumps(ai_result["keywords"], ensure_ascii=False),
        ai_photo_analysis=photo_analysis,
        photo_path=photo_path,
        photo_paths_json=json.dumps(photo_paths, ensure_ascii=False) if photo_paths else None,
        status="open"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    
    # 5. Créer message
    message = Message(
        ticket_id=ticket.id,
        sender_type="user",
        sender_id=user.id,
        content=data.content,
        channel=data.channel
    )
    db.add(message)
    db.commit()
    
    # 6. Retourner résultat avec analyse IA COMPLÈTE
    response = {
        "status": "success",
        "ticket_id": ticket.id,
        "ai_analysis": ai_result,
        "knowledge_mode": knowledge_result["knowledge_mode"],
        "knowledge_fallback_used": knowledge_result["knowledge_fallback_used"],
    }
    
    # Ajouter l'analyse photo si disponible
    if photo_analysis:
        try:
            response["photo_analysis"] = photo_analysis_payload or json.loads(photo_analysis)
        except:
            response["photo_analysis"] = {"analysis": "Analyse photo en cours"}
    
    # Ajouter l'URL de la photo
    if photo_path:
        response["photo_url"] = f"http://localhost:8000/{photo_path}"
    if photo_paths:
        response["photo_urls"] = [_build_upload_url(path) for path in photo_paths]

    # Toujours retourner les fiches RAG utilisées (pour debug et fallback côté frontend)
    if rag_items:
        response["rag_items"] = rag_items

    # Ajouter la réponse principale générée par le LLM si disponible
    if llm_answer:
        response["llm_answer"] = llm_answer
    elif knowledge_result["rag_fallback_answer"]:
        # Fallback déterministe : utiliser la réponse de la meilleure fiche
        # pour que l'utilisateur ait au moins la réponse validée locale,
        # même si la clé OpenAI n'est pas configurée.
        response["rag_fallback_answer"] = knowledge_result["rag_fallback_answer"]
    
    return response


@app.post("/api/assistant/query")
async def assistant_query(data: MessageCreate, db: Session = Depends(get_db)):
    """Endpoint conversation IA seule (RAG + GPT) sans création de ticket.

    Utilisé par l'application pour discuter avec l'IA et affiner le problème.
    Aucun Ticket/Message n'est créé ici, uniquement une réponse IA.
    """

    # 1. Analyse IA texte
    ai_result = ai_engine.classify(data.content)

    # 2. Déterminer la catégorie finale et le domaine RAG
    nlp_category = ai_result.get("category", "agriculture")
    chosen_category = data.category or nlp_category
    ai_result["classifier_category"] = nlp_category
    ai_result["category"] = chosen_category

    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"

    # 3. Analyse photo si présente (sans sauvegarder de fichier)
    photo_analysis = None
    photo_payloads = _collect_photo_payloads(data.photo_base64, data.photo_base64_list)
    if photo_payloads:
        try:
            photo_data_list = [_decode_photo_payload(payload) for payload in photo_payloads]
            photo_analysis_result = cv_engine.analyze_images(photo_data_list, data.content, data.category)
            photo_analysis = photo_analysis_result

            if photo_analysis_result.get("urgency") == "high":
                ai_result["urgency"] = "high"
        except Exception as e:
            print(f"Erreur analyse photo (assistant_query): {e}")
            photo_analysis = {"error": str(e), "requires_expert": True}

    # 4. Construire le contexte conversationnel et enrichir la recherche RAG
    conversation_context = []
    if data.conversation_context:
        conversation_context = [
            {"role": turn.role, "content": turn.content}
            for turn in data.conversation_context
            if (turn.content or "").strip()
        ]

    contextual_query_parts = [
        turn["content"]
        for turn in conversation_context[-4:]
    ]
    contextual_query_parts.append(data.content)
    contextual_query = "\n".join(part for part in contextual_query_parts if part)

    # 5. RAG strict d'abord, puis base élargie seulement si besoin
    focus_context = extract_focus_context(
        chosen_category,
        contextual_query,
        photo_analysis,
    )

    knowledge_result = resolve_knowledge_answer(
        db=db,
        domain=kb_domain,
        question=contextual_query,
        language="fr",
        conversation_context=conversation_context,
        focus_context=focus_context,
    )
    rag_items = knowledge_result["rag_items"]
    llm_answer = knowledge_result["llm_answer"]

    # Log the knowledge source
    try:
        knowledge_mode = knowledge_result['knowledge_mode']
        if knowledge_mode == "rag_strict":
            print(f"✓ [ASSISTANT] RAG-STRICT | {len(rag_items)} fiches trouvées")
        elif knowledge_mode == "rag_expanded":
            print(f"⚠ [ASSISTANT] RAG-EXPANDED | {len(rag_items)} fiches (recherche élargie)")
        elif knowledge_mode == "llm_general_knowledge":
            print(f"⚡ [ASSISTANT] LLM-GENERAL | Connaissances générales sans RAG")
        elif knowledge_mode == "no_match":
            print(f"✗ [ASSISTANT] NO-MATCH | Aucune source disponible")
    except Exception as e:
        print(f"[ASSISTANT] Log error: {e}")

    # 6. Construction de la réponse (sans ticket)
    response: Dict[str, Any] = {
        "status": "success",
        "ai_analysis": ai_result,
        "category": chosen_category,
        "knowledge_mode": knowledge_result["knowledge_mode"],
        "knowledge_fallback_used": knowledge_result["knowledge_fallback_used"],
    }

    if photo_analysis is not None:
        response["photo_analysis"] = photo_analysis

    if rag_items:
        response["rag_items"] = rag_items

    if llm_answer:
        response["llm_answer"] = llm_answer
    elif knowledge_result["rag_fallback_answer"]:
        response["rag_fallback_answer"] = knowledge_result["rag_fallback_answer"]

    return response

@app.post("/api/tickets/{ticket_id}/reply")
async def reply_to_ticket(
    ticket_id: int, 
    content: ReplyMessage,
    db: Session = Depends(get_db)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    message = Message(
        ticket_id=ticket_id,
        sender_type="expert",
        sender_id=1,
        content=content.message,
        channel="web"
    )
    db.add(message)
    
    if not ticket.expert_id:
        ticket.expert_id = 1
        ticket.status = "assigned"
    
    db.commit()
    return {"status": "success"}

@app.post("/api/tickets/{ticket_id}/send-to-expert")
async def send_ticket_to_expert(
    ticket_id: int,
    db: Session = Depends(get_db)
):
    """Envoie un ticket à un expert humain (enregistre la demande)."""
    
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Assigner à un expert (ID 1 par défaut, ou le premier expert disponible)
    if not ticket.expert_id:
        expert = db.query(Expert).filter(Expert.is_active == True).first()
        if expert:
            ticket.expert_id = expert.id
        else:
            ticket.expert_id = 1  # Fallback
    
    # Marquer comme "awaiting_expert"
    ticket.status = "awaiting_expert"
    
    # Créer un message système pour tracer
    message = Message(
        ticket_id=ticket_id,
        sender_type="system",
        sender_id=None,
        content="[SYSTEME] Aide d'expert demandee par l'utilisateur",
        channel="app"
    )
    db.add(message)
    db.commit()
    db.refresh(ticket)
    
    return {
        "status": "success",
        "ticket_id": ticket.id,
        "expert_assigned": ticket.expert_id,
        "message": "Votre demande a ete envoyee a un expert. Un expert humain vous contactera bientot."
    }

@app.post("/api/tickets/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    ticket.status = "resolved"
    ticket.resolved_at = datetime.utcnow()
    
    message = Message(
        ticket_id=ticket_id,
        sender_type="system",
        sender_id=None,
        content="Ticket marqué comme résolu par l'expert",
        channel="system"
    )
    db.add(message)
    
    db.commit()
    return {"status": "success"}


@app.get("/api/tickets/{ticket_id}/ai-summary")
async def get_ticket_ai_summary(ticket_id: int, db: Session = Depends(get_db)):
    """Retourne un résumé IA (RAG + GPT ou fallback) pour un ticket donné.

    L'IA se base sur le dernier message utilisateur du ticket et sur
    la catégorie du ticket pour choisir le bon domaine de connaissance.
    """

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Récupérer le dernier message utilisateur lié au ticket
    last_user_msg = (
        db.query(Message)
        .filter(Message.ticket_id == ticket_id, Message.sender_type == "user")
        .order_by(Message.sent_at.desc())
        .first()
    )

    if not last_user_msg:
        return {"status": "success", "ai_summary": None, "detail": "Aucun message utilisateur pour ce ticket."}

    content = last_user_msg.content or ""

    # Mapper la catégorie du ticket vers le domaine RAG
    chosen_category = ticket.category or "agriculture"
    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"

    focus_context = extract_focus_context(
        chosen_category,
        content,
    )

    knowledge_result = resolve_knowledge_answer(
        db=db,
        domain=kb_domain,
        question=content,
        language="fr",
        focus_context=focus_context,
    )
    rag_items = knowledge_result["rag_items"]
    llm_answer = knowledge_result["llm_answer"] or knowledge_result["rag_fallback_answer"]

    return {
        "status": "success",
        "ai_summary": llm_answer,
        "category": chosen_category,
        "rag_items": rag_items,
        "knowledge_mode": knowledge_result["knowledge_mode"],
        "knowledge_fallback_used": knowledge_result["knowledge_fallback_used"],
    }

@app.get("/api/user-tickets")
async def get_user_tickets(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        return []
    
    tickets = db.query(Ticket).filter(
        Ticket.user_id == user.id
    ).order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        last_msg = db.query(Message).filter(
            Message.ticket_id == ticket.id
        ).order_by(Message.sent_at.desc()).first()
        
        # Construire l'URL de la photo
        photo_url = None
        if ticket.photo_path:
            photo_url = f"http://localhost:8000/{ticket.photo_path}"
        photo_paths = _load_json_list(ticket.photo_paths_json)
        photo_urls = [_build_upload_url(path) for path in photo_paths if path]
        
        result.append({
            "id": ticket.id,
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else "Aucun message",
            "has_photo": ticket.photo_path is not None,
            "photo_url": photo_url,
            "photo_urls": photo_urls,
        })
    
    return result


@app.get("/api/photo-analyses")
async def get_photo_analyses(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        return []

    records = (
        db.query(PhotoAnalysisHistoryDB)
        .filter(PhotoAnalysisHistoryDB.user_id == user.id)
        .order_by(PhotoAnalysisHistoryDB.created_at.desc())
        .limit(50)
        .all()
    )
    return [_serialize_photo_history_record(record) for record in records]


@app.post("/api/photo-analyses")
async def save_photo_analysis_history(
    data: PhotoAnalysisHistoryIn,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.phone_number == data.phone_number.strip()).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    record = None
    if data.client_record_id:
        record = (
            db.query(PhotoAnalysisHistoryDB)
            .filter(
                PhotoAnalysisHistoryDB.user_id == user.id,
                PhotoAnalysisHistoryDB.client_record_id == data.client_record_id,
            )
            .first()
        )

    photo_data_list = [
        _decode_photo_payload(payload)
        for payload in data.photo_base64_list[:3]
        if payload
    ]
    stored_paths = (
        _store_photo_payloads(user.id, photo_data_list, prefix="analysis")
        if photo_data_list
        else []
    )

    if record is None:
        record = PhotoAnalysisHistoryDB(
            user_id=user.id,
            client_record_id=data.client_record_id,
            category=data.category,
            prompt=data.prompt,
            analysis_json=json.dumps(data.analysis, ensure_ascii=False),
            photo_paths_json=json.dumps(stored_paths, ensure_ascii=False),
            photo_labels_json=json.dumps(
                data.photo_labels[: len(stored_paths)],
                ensure_ascii=False,
            ),
        )
        db.add(record)
    else:
        record.category = data.category
        record.prompt = data.prompt
        record.analysis_json = json.dumps(data.analysis, ensure_ascii=False)
        if stored_paths:
            record.photo_paths_json = json.dumps(stored_paths, ensure_ascii=False)
            record.photo_labels_json = json.dumps(
                data.photo_labels[: len(stored_paths)],
                ensure_ascii=False,
            )

    db.commit()
    db.refresh(record)
    return _serialize_photo_history_record(record)


@app.get("/api/knowledge/offline-cache")
async def knowledge_offline_cache(
    db: Session = Depends(get_db),
):
    """
    Base de connaissances formatée pour mise en cache offline côté app.
    Retourne tous les items actifs avec les champs nécessaires au RAG local.
    """
    items = db.query(KnowledgeItem).order_by(KnowledgeItem.domain).all()
    result = []
    for it in items:
        result.append({
            "id": it.id,
            "domain": it.domain,
            "title": it.title,
            "question": it.question,
            "answer": it.answer,
            "tags": _load_json_list(it.tags),
            "language": it.language or "fr",
        })
    return {"items": result, "total": len(result)}


@app.get("/api/admin/knowledge")
async def list_knowledge_items(
    domain: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Lister les fiches de la base de connaissances.

    Pour l'instant, cette route n'est pas protégée (usage local / démo). Pour
    un déploiement réel, il faudra ajouter une authentification admin.
    """
    query = db.query(KnowledgeItem)
    if domain:
        query = query.filter(KnowledgeItem.domain == domain)

    items = query.order_by(KnowledgeItem.created_at.desc()).all()

    result: List[Dict[str, Any]] = []
    for it in items:
        media_data = None
        if it.media:
            try:
                media_data = json.loads(it.media)
            except Exception:
                media_data = None

        result.append(
            {
                "id": it.id,
                "domain": it.domain,
                "title": it.title,
                "question": it.question,
                "answer": it.answer,
                "tags": _load_json_list(it.tags),
                "language": it.language,
                "source": it.source,
                "media": media_data,
                "created_at": it.created_at,
                "updated_at": it.updated_at,
            }
        )

    return result


@app.get("/api/emergency-numbers")
async def public_emergency_numbers(db: Session = Depends(get_db)):
    """Liste publique des numéros d'urgence / numéros utiles.

    Utilisée par l'application utilisateur pour afficher les numéros utiles.
    """
    items = db.query(EmergencyNumber).filter(EmergencyNumber.is_active == True).order_by(
        EmergencyNumber.display_order.asc(), EmergencyNumber.id.asc()
    ).all()

    return [
        {
            "id": it.id,
            "label": it.label,
            "number": it.number,
            "description": it.description,
            "display_order": it.display_order,
        }
        for it in items
    ]


@app.post("/api/admin/reload-knowledge")
async def reload_knowledge_endpoint(
    db: Session = Depends(get_db),
):
    """Recharger la base de connaissances depuis le fichier JSON.

    End-point simple pour éviter de redémarrer le serveur quand tu mets à jour
    knowledge_base.json. Dans un vrai déploiement, il faudra protéger cette route
    (token admin, VPN, etc.).
    """
    load_knowledge_from_json(db)
    total_items = db.query(KnowledgeItem).count()
    return {
        "status": "success",
        "total_items": total_items,
    }


@app.post("/api/admin/knowledge")
async def create_knowledge_item(
    payload: KnowledgeItemIn,
    db: Session = Depends(get_db),
):
    """Créer une nouvelle fiche de connaissance (usage panneau expert)."""

    media_json = None
    if payload.media:
        media_json = json.dumps(
            [m.dict() for m in payload.media], ensure_ascii=False
        )

    item = KnowledgeItem(
        domain=payload.domain,
        title=payload.title,
        question=payload.question,
        answer=payload.answer,
        tags=json.dumps(payload.tags or [], ensure_ascii=False),
        language=payload.language,
        source=payload.source,
        media=media_json,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    media_data = None
    if item.media:
        try:
            media_data = json.loads(item.media)
        except Exception:
            media_data = None

    return {
        "id": item.id,
        "domain": item.domain,
        "title": item.title,
        "question": item.question,
        "answer": item.answer,
        "tags": json.loads(item.tags) if item.tags else [],
        "language": item.language,
        "source": item.source,
        "media": media_data,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@app.get("/api/admin/emergency-numbers")
async def list_emergency_numbers_admin(db: Session = Depends(get_db)):
    """Lister tous les numéros utiles (admin panneau expert)."""

    items = db.query(EmergencyNumber).order_by(
        EmergencyNumber.display_order.asc(), EmergencyNumber.id.asc()
    ).all()

    return [
        {
            "id": it.id,
            "label": it.label,
            "number": it.number,
            "description": it.description,
            "display_order": it.display_order,
            "is_active": it.is_active,
        }
        for it in items
    ]


@app.post("/api/admin/emergency-numbers")
async def create_emergency_number(
    payload: EmergencyNumberIn,
    db: Session = Depends(get_db),
):
    """Créer un nouveau numéro utile (pompier, police, clinique...)."""

    item = EmergencyNumber(
        label=payload.label,
        number=payload.number,
        description=payload.description,
        display_order=payload.display_order,
        is_active=True,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "id": item.id,
        "label": item.label,
        "number": item.number,
        "description": item.description,
        "display_order": item.display_order,
        "is_active": item.is_active,
    }


@app.put("/api/admin/knowledge/{item_id}")
async def update_knowledge_item(
    item_id: int,
    payload: KnowledgeItemIn,
    db: Session = Depends(get_db),
):
    """Mettre à jour une fiche de connaissance existante ou la créer si elle n'existe plus.

    Cela évite une erreur 404 côté panneau expert si, pour une raison quelconque,
    l'ID stocké dans le frontend ne correspond plus à une ligne en base.
    """

    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()

    media_json = None
    if payload.media:
        media_json = json.dumps(
            [m.dict() for m in payload.media], ensure_ascii=False
        )

    if not item:
        # Comportement "upsert" : si l'ID n'existe plus, on crée une nouvelle fiche
        item = KnowledgeItem(
            domain=payload.domain,
            title=payload.title,
            question=payload.question,
            answer=payload.answer,
            tags=json.dumps(payload.tags or [], ensure_ascii=False),
            language=payload.language,
            source=payload.source,
            media=media_json,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
    else:
        item.domain = payload.domain
        item.title = payload.title
        item.question = payload.question
        item.answer = payload.answer
        item.tags = json.dumps(payload.tags or [], ensure_ascii=False)
        item.language = payload.language
        item.source = payload.source
        item.media = media_json

        db.commit()
        db.refresh(item)

    media_data = None
    if item.media:
        try:
            media_data = json.loads(item.media)
        except Exception:
            media_data = None

    return {
        "id": item.id,
        "domain": item.domain,
        "title": item.title,
        "question": item.question,
        "answer": item.answer,
        "tags": json.loads(item.tags) if item.tags else [],
        "language": item.language,
        "source": item.source,
        "media": media_data,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@app.delete("/api/admin/knowledge/{item_id}")
async def delete_knowledge_item(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Supprimer une fiche de connaissance."""

    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")

    db.delete(item)
    db.commit()
    return {"status": "success"}


@app.put("/api/admin/emergency-numbers/{item_id}")
async def update_emergency_number(
    item_id: int,
    payload: EmergencyNumberIn,
    db: Session = Depends(get_db),
):
    """Mettre à jour un numéro utile existant."""

    item = db.query(EmergencyNumber).filter(EmergencyNumber.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Emergency number not found")

    item.label = payload.label
    item.number = payload.number
    item.description = payload.description
    item.display_order = payload.display_order

    db.commit()
    db.refresh(item)

    return {
        "id": item.id,
        "label": item.label,
        "number": item.number,
        "description": item.description,
        "display_order": item.display_order,
        "is_active": item.is_active,
    }


@app.delete("/api/admin/emergency-numbers/{item_id}")
async def delete_emergency_number(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Supprimer un numéro utile."""

    item = db.query(EmergencyNumber).filter(EmergencyNumber.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Emergency number not found")

    db.delete(item)
    db.commit()
    return {"status": "success"}


@app.post("/api/admin/knowledge/import-json")
async def import_knowledge_from_json(
    payload: KnowledgeBulkImport,
    db: Session = Depends(get_db),
):
    """Importer/mettre à jour des fiches à partir d'un JSON envoyé par le panneau expert.

    Les fiches sont identifiées par (domain, title). Si une fiche existe déjà,
    elle est mise à jour; sinon elle est créée.
    """

    created = 0
    updated = 0

    for entry in payload.items:
        before = db.query(KnowledgeItem).filter(KnowledgeItem.title == entry.title).count()
        _upsert_knowledge_item(
            db,
            domain=entry.domain,
            title=entry.title,
            question=entry.question,
            answer=entry.answer,
            tags=entry.tags or [],
            language=entry.language,
            source=entry.source,
            media=[m.dict() for m in entry.media] if entry.media else None,
        )
        after = db.query(KnowledgeItem).filter(KnowledgeItem.title == entry.title).count()
        if before == 0 and after == 1:
            created += 1
        else:
            updated += 1

    db.commit()

    total_items = db.query(KnowledgeItem).count()
    return {
        "status": "success",
        "created": created,
        "updated": updated,
        "total_items": total_items,
    }

@app.post("/api/create-test-expert")
async def create_test_expert(db: Session = Depends(get_db)):
    existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
    if existing:
        return {
            "message": "Expert déjà existant", 
            "email": "test@resolvehub.bf", 
            "password": "test123"
        }
    
    expert = Expert(
        email="test@resolvehub.bf",
        password_hash=hash_password("test123"),
        full_name="Expert Test IA",
        specialization="agriculture",
        is_active=True
    )
    db.add(expert)
    db.commit()
    
    return {
        "message": "Expert créé avec succès", 
        "email": "test@resolvehub.bf", 
        "password": "test123"
    }

# ==========================================
# NOUVEAUX ENDPOINTS POUR FONCTIONNALITÉS PRIORITAIRES
# ==========================================

class SOSAlert(BaseModel):
    phoneNumber: str
    type: str  # accident, attack, fire, animal_sick, community
    description: Optional[str] = None
    location: Dict[str, Any]  # {latitude, longitude, accuracy, note}
    timestamp: int
    urgent: bool = True

class SOSAlertDB(Base):
    __tablename__ = "sos_alerts"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    location_note = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, acknowledged, resolved
    notified_authorities = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessageDB(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # numéro ou nom affiché
    text = Column(Text, nullable=False)
    is_bot = Column(Boolean, default=False)
    room = Column(String, default="general")
    sender_role = Column(String, default="member")
    report_count = Column(Integer, default=0)
    is_hidden = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    pinned_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)


def _ensure_sos_description_column() -> None:
    """Migration: ajouter la colonne 'description' à sos_alerts si absente."""
    if not str(engine.url).startswith("sqlite"):
        return
    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA table_info(sos_alerts)")
            columns = [row[1] for row in result]
            if "description" not in columns:
                conn.exec_driver_sql("ALTER TABLE sos_alerts ADD COLUMN description TEXT")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter la colonne 'description' à sos_alerts: {e}")

_ensure_sos_description_column()


def _ensure_chat_message_columns() -> None:
    """Migration légère pour enrichir le module communauté."""
    if not str(engine.url).startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA table_info(chat_messages)")
            columns = [row[1] for row in result]

            if "room" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN room TEXT DEFAULT 'general'"
                )
            if "sender_role" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN sender_role TEXT DEFAULT 'member'"
                )
            if "report_count" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN report_count INTEGER DEFAULT 0"
                )
            if "is_hidden" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN is_hidden BOOLEAN DEFAULT 0"
                )
            if "is_pinned" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN is_pinned BOOLEAN DEFAULT 0"
                )
            if "pinned_at" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN pinned_at DATETIME"
                )
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter les colonnes de communauté: {e}")


_ensure_chat_message_columns()

@app.post("/api/sos/alert")
async def create_sos_alert(alert: SOSAlert, db: Session = Depends(get_db)):
    """
    🔑 ENDPOINT SOS - PRIORITÉ CRITIQUE
    Enregistrer une alerte SOS avec géolocalisation
    """
    try:
        # Créer l'alerte SOS
        sos_alert = SOSAlertDB(
            phone_number=alert.phoneNumber or "Anonyme",
            alert_type=alert.type,
            description=alert.description,
            latitude=alert.location.get("latitude"),
            longitude=alert.location.get("longitude"),
            accuracy=alert.location.get("accuracy"),
            location_note=alert.location.get("note"),
            status="pending",
            notified_authorities=False
        )
        
        db.add(sos_alert)
        db.commit()
        db.refresh(sos_alert)
        
        # TODO: Notifier les autorités locales / ONG partenaires
        # - Envoi SMS aux numéros d'urgence
        # - Notification email aux responsables
        # - Webhook vers systèmes externes
        
        print(f"🚨 ALERTE SOS REÇUE - Type: {alert.type}, Tel: {alert.phoneNumber}")
        if alert.location.get("latitude"):
            print(f"   📍 Position: {alert.location.get('latitude')}, {alert.location.get('longitude')}")
        
        return {
            "success": True,
            "alert_id": sos_alert.id,
            "message": "Alerte SOS enregistrée. Secours notifiés.",
            "emergency_numbers": [
                {"label": "Police", "number": "17"},
                {"label": "Pompiers", "number": "18"},
                {"label": "SAMU", "number": "15"}
            ]
        }
    
    except Exception as e:
        print(f"❌ Erreur SOS: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur enregistrement SOS: {str(e)}")

@app.patch("/api/sos/alerts/{alert_id}/status")
async def update_sos_alert_status(
    alert_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Mettre à jour le statut d'une alerte SOS (pour panel expert).
    Body JSON: { "status": "acknowledged" | "resolved" }
    """
    alert = db.query(SOSAlertDB).filter(SOSAlertDB.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte SOS introuvable")
    new_status = body.get("status")
    if new_status not in ("pending", "acknowledged", "resolved"):
        raise HTTPException(status_code=422, detail="Statut invalide")
    alert.status = new_status
    db.commit()
    return {"success": True, "alert_id": alert_id, "status": new_status}

@app.get("/api/sos/alerts")
async def get_sos_alerts(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """
    Récupérer les alertes SOS (pour dashboard admin/expert)
    """
    query = db.query(SOSAlertDB).order_by(SOSAlertDB.created_at.desc())
    
    if status:
        query = query.filter(SOSAlertDB.status == status)
    
    alerts = query.limit(limit).all()
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "phoneNumber": alert.phone_number,
                "type": alert.alert_type,
                "description": alert.description,
                "location": {
                    "latitude": alert.latitude,
                    "longitude": alert.longitude,
                    "accuracy": alert.accuracy,
                    "note": alert.location_note
                },
                "status": alert.status,
                "timestamp": alert.created_at.isoformat() if alert.created_at else None
            }
            for alert in alerts
        ],
        "total": len(alerts)
    }


# ==========================================
# COMMUNITY CHAT
# ==========================================

BOT_REPLIES = [
    "Merci pour votre partage ! Un expert vous répondra bientôt.",
    "Bonne question ! D'autres membres ont eu le même problème.",
    "Consultez aussi la section Agriculture pour plus de conseils.",
    "Vous n'êtes pas seul ! La communauté SONGRA est là pour vous aider.",
    "Votre question a bien été notée. Restez connecté !",
]

COMMUNITY_ROOM_LABELS = {
    "general": "Général",
    "agriculture": "Agriculture",
    "elevage": "Élevage",
    "securite": "Sécurité",
    "marche": "Marché",
}

COMMUNITY_ROOM_TO_DOMAIN = {
    "general": "agriculture",
    "agriculture": "agriculture",
    "elevage": "elevage",
    "securite": "cybersecurity",
    "marche": "agriculture",
}

COMMUNITY_DUPLICATE_WINDOW_SECONDS = 120
COMMUNITY_REPORT_HIDE_THRESHOLD = 3
COMMUNITY_BLOCKED_TERMS = {
    "arnaque",
    "escroc",
    "haine",
    "insulte",
}


def _normalize_community_room(raw_room: Optional[str]) -> str:
    normalized = (raw_room or "general").strip().lower()
    return normalized if normalized in COMMUNITY_ROOM_LABELS else "general"


def _serialize_community_message(message: ChatMessageDB) -> Dict[str, Any]:
    normalized_room = _normalize_community_room(message.room)
    sender_role = message.sender_role or ("assistant" if message.is_bot else "member")
    return {
        "id": message.id,
        "sender": message.sender,
        "text": message.text,
        "is_bot": message.is_bot,
        "created_at": message.created_at.isoformat() if message.created_at else None,
        "room": normalized_room,
        "room_label": COMMUNITY_ROOM_LABELS.get(normalized_room, "Général"),
        "sender_role": sender_role,
        "is_expert": sender_role == "expert",
        "report_count": message.report_count or 0,
        "is_pinned": bool(message.is_pinned),
        "pinned_at": message.pinned_at.isoformat() if message.pinned_at else None,
    }


def _get_pinned_community_message(db: Session, room: str) -> Optional[ChatMessageDB]:
    return (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.room == room)
        .filter(ChatMessageDB.is_hidden == False)
        .filter(ChatMessageDB.is_pinned == True)
        .order_by(ChatMessageDB.pinned_at.desc(), ChatMessageDB.id.desc())
        .first()
    )


def _set_pinned_community_message(
    db: Session,
    message: ChatMessageDB,
    pinned: bool,
) -> ChatMessageDB:
    db.query(ChatMessageDB).filter(ChatMessageDB.room == message.room).update(
        {ChatMessageDB.is_pinned: False, ChatMessageDB.pinned_at: None},
        synchronize_session=False,
    )

    if pinned:
        message.is_pinned = True
        message.pinned_at = datetime.utcnow()
    else:
        message.is_pinned = False
        message.pinned_at = None

    db.commit()
    db.refresh(message)
    return message


def _contains_blocked_community_text(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in COMMUNITY_BLOCKED_TERMS)


def _build_contextual_bot_reply(db: Session, room: str, text: str) -> str:
    domain = COMMUNITY_ROOM_TO_DOMAIN.get(room, "agriculture")
    items = retrieve_knowledge(db, domain, text, limit=2)
    room_label = COMMUNITY_ROOM_LABELS.get(room, "Général")

    if items:
        best = items[0]
        title = best.get("title") or room_label
        answer = (best.get("answer") or "").replace("\n", " ").strip()
        if len(answer) > 220:
            answer = answer[:220].rsplit(" ", 1)[0] + "..."

        if room == "marche":
            return (
                f"Repère marché: {title}. {answer} "
                "Compare aussi les prix locaux avant de vendre ou d'acheter."
            )

        if room == "securite":
            return (
                f"Point sécurité: {title}. {answer} "
                "Ne partage jamais ton code ou ton mot de passe par message."
            )

        return f"Conseil {room_label.lower()}: {title}. {answer}"

    lowered = text.lower()
    if any(keyword in lowered for keyword in ["urgent", "grave", "mort", "attaque"]):
        return (
            "La situation semble sérieuse. Décris le lieu, depuis quand le problème a commencé, "
            "et contacte aussi un expert ou le module SOS si nécessaire."
        )

    if room == "marche":
        return (
            "Précise le produit, la quantité, le prix observé et la commune. "
            "La communauté pourra comparer plus facilement."
        )

    return BOT_REPLIES[hash(f"{room}:{text[:40]}") % len(BOT_REPLIES)]

@app.get("/api/community/messages")
async def get_community_messages(
    room: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """Récupérer les derniers messages du chat communautaire."""
    query = db.query(ChatMessageDB).filter(ChatMessageDB.is_hidden == False)
    if room is not None:
        query = query.filter(ChatMessageDB.room == _normalize_community_room(room))

    messages = query.order_by(
        ChatMessageDB.created_at.desc(), ChatMessageDB.id.desc()
    ).limit(limit).all()
    messages.reverse()
    return [_serialize_community_message(m) for m in messages]


@app.get("/api/community/pinned")
async def get_pinned_community_message(
    room: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Récupérer l'annonce experte épinglée pour un salon."""
    normalized_room = _normalize_community_room(room)
    message = _get_pinned_community_message(db, normalized_room)
    return {
        "room": normalized_room,
        "message": _serialize_community_message(message) if message else None,
    }

@app.post("/api/community/messages")
async def post_community_message(
    body: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Poster un message dans le chat communautaire et générer une réponse bot."""
    sender = (body.get("sender") or "Anonyme")[:80]
    text = (body.get("text") or "").strip()
    room = _normalize_community_room(body.get("room"))
    sender_role = (body.get("sender_role") or "member").strip().lower()
    if sender_role not in {"member", "expert", "assistant"}:
        sender_role = "member"
    if not text:
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide")
    if len(text) > 1000:
        raise HTTPException(status_code=422, detail="Message trop long (max 1000 caractères)")
    if _contains_blocked_community_text(text):
        raise HTTPException(status_code=422, detail="Message bloqué par la modération légère de la communauté")

    duplicate_since = datetime.utcnow() - timedelta(seconds=COMMUNITY_DUPLICATE_WINDOW_SECONDS)
    duplicate = (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.sender == sender)
        .filter(ChatMessageDB.text == text)
        .filter(ChatMessageDB.room == room)
        .filter(ChatMessageDB.created_at >= duplicate_since)
        .filter(ChatMessageDB.is_hidden == False)
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="Message déjà publié récemment dans ce salon")

    user_msg = ChatMessageDB(
        sender=sender,
        text=text,
        is_bot=False,
        room=room,
        sender_role=sender_role,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    bot_text = _build_contextual_bot_reply(db, room, text)
    bot_msg = ChatMessageDB(
        sender="Assistant SONGRA",
        text=bot_text,
        is_bot=True,
        room=room,
        sender_role="assistant",
    )
    db.add(bot_msg)
    db.commit()
    db.refresh(bot_msg)

    return {
        "user": _serialize_community_message(user_msg),
        "bot": _serialize_community_message(bot_msg),
    }


@app.post("/api/community/messages/{message_id}/report")
async def report_community_message(
    message_id: int,
    db: Session = Depends(get_db)
):
    """Signaler un message communautaire. Auto-masquage après plusieurs signalements."""
    message = db.query(ChatMessageDB).filter(ChatMessageDB.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message introuvable")

    if message.is_hidden:
        return {
            "success": True,
            "message_id": message_id,
            "report_count": message.report_count or 0,
            "status": "hidden",
        }

    message.report_count = (message.report_count or 0) + 1
    if message.report_count >= COMMUNITY_REPORT_HIDE_THRESHOLD:
        message.is_hidden = True
    db.commit()
    db.refresh(message)

    return {
        "success": True,
        "message_id": message.id,
        "report_count": message.report_count,
        "status": "hidden" if message.is_hidden else "reported",
    }


@app.post("/api/expert/community/messages")
async def post_expert_community_message(
    body: Dict[str, Any],
    current_expert: Expert = Depends(get_current_expert),
    db: Session = Depends(get_db),
):
    """Permet à un expert connecté de publier dans un salon communautaire."""
    text = (body.get("text") or "").strip()
    room = _normalize_community_room(body.get("room"))
    pin_message = bool(body.get("pin"))
    sender = (current_expert.full_name or current_expert.email or "Expert SONGRA").strip()[:80]

    if not text:
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide")
    if len(text) > 1000:
        raise HTTPException(status_code=422, detail="Message trop long (max 1000 caractères)")
    if _contains_blocked_community_text(text):
        raise HTTPException(status_code=422, detail="Message bloqué par la modération légère de la communauté")

    duplicate_since = datetime.utcnow() - timedelta(seconds=COMMUNITY_DUPLICATE_WINDOW_SECONDS)
    duplicate = (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.sender == sender)
        .filter(ChatMessageDB.text == text)
        .filter(ChatMessageDB.room == room)
        .filter(ChatMessageDB.created_at >= duplicate_since)
        .filter(ChatMessageDB.is_hidden == False)
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="Message déjà publié récemment dans ce salon")

    expert_message = ChatMessageDB(
        sender=sender,
        text=text,
        is_bot=False,
        room=room,
        sender_role="expert",
    )
    db.add(expert_message)
    db.commit()
    db.refresh(expert_message)

    if pin_message:
        expert_message = _set_pinned_community_message(db, expert_message, True)

    return {
        "success": True,
        "message": _serialize_community_message(expert_message),
    }


@app.patch("/api/expert/community/messages/{message_id}/pin")
async def pin_expert_community_message(
    message_id: int,
    body: Dict[str, Any],
    current_expert: Expert = Depends(get_current_expert),
    db: Session = Depends(get_db),
):
    """Épingler ou retirer une annonce experte dans un salon communautaire."""
    del current_expert
    pinned = bool(body.get("pinned", True))
    message = db.query(ChatMessageDB).filter(ChatMessageDB.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message introuvable")
    if message.is_hidden:
        raise HTTPException(status_code=422, detail="Impossible d'épingler un message masqué")
    if (message.sender_role or "") != "expert":
        raise HTTPException(status_code=422, detail="Seuls les messages experts peuvent être épinglés")

    updated_message = _set_pinned_community_message(db, message, pinned)
    return {
        "success": True,
        "message": _serialize_community_message(updated_message),
        "pinned": bool(updated_message.is_pinned),
    }


@app.patch("/api/expert/community/messages/{message_id}")
async def update_expert_community_message(
    message_id: int,
    body: Dict[str, Any],
    current_expert: Expert = Depends(get_current_expert),
    db: Session = Depends(get_db),
):
    """Modifier directement un message expert communautaire, notamment une annonce épinglée."""
    del current_expert
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide")
    if len(text) > 1000:
        raise HTTPException(status_code=422, detail="Message trop long (max 1000 caractères)")
    if _contains_blocked_community_text(text):
        raise HTTPException(status_code=422, detail="Message bloqué par la modération légère de la communauté")

    message = db.query(ChatMessageDB).filter(ChatMessageDB.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message introuvable")
    if message.is_hidden:
        raise HTTPException(status_code=422, detail="Impossible de modifier un message masqué")
    if (message.sender_role or "") != "expert":
        raise HTTPException(status_code=422, detail="Seuls les messages experts peuvent être modifiés")

    message.text = text
    db.commit()
    db.refresh(message)

    return {
        "success": True,
        "message": _serialize_community_message(message),
    }

class OfflineSyncData(BaseModel):
    tickets: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []
    photos: List[Dict[str, Any]] = []

@app.post("/api/sync/offline")
async def sync_offline_data(sync_data: OfflineSyncData, db: Session = Depends(get_db)):
    """
    🔑 ENDPOINT SYNCHRONISATION OFFLINE
    Synchroniser les données sauvegardées localement quand connexion revient
    """
    synced_tickets = []
    synced_messages = []
    synced_photos = []
    errors = []
    
    try:
        # Synchroniser les tickets
        for ticket_data in sync_data.tickets:
            try:
                # Vérifier si l'utilisateur existe
                phone = ticket_data.get("phoneNumber", "")
                user = db.query(User).filter(User.phone_number == phone).first()
                
                if not user:
                    user = User(phone_number=phone, name=ticket_data.get("userName"))
                    db.add(user)
                    db.commit()
                    db.refresh(user)
                
                # Créer le ticket
                ticket = Ticket(
                    user_id=user.id,
                    category=ticket_data.get("category"),
                    urgency=ticket_data.get("urgency", "medium"),
                    status="open"
                )
                
                db.add(ticket)
                db.commit()
                db.refresh(ticket)
                
                synced_tickets.append({
                    "localId": ticket_data.get("localId"),
                    "serverId": ticket.id
                })
                
            except Exception as e:
                errors.append({"type": "ticket", "error": str(e), "data": ticket_data.get("localId")})
        
        # Synchroniser les messages
        for msg_data in sync_data.messages:
            try:
                message = Message(
                    ticket_id=msg_data.get("ticketId"),
                    sender_type="user",
                    sender_id=msg_data.get("userId"),
                    content=msg_data.get("content"),
                    channel="app"
                )
                
                db.add(message)
                db.commit()
                
                synced_messages.append({
                    "localId": msg_data.get("localId"),
                    "serverId": message.id
                })
                
            except Exception as e:
                errors.append({"type": "message", "error": str(e)})
        
        return {
            "success": True,
            "synced": {
                "tickets": len(synced_tickets),
                "messages": len(synced_messages),
                "photos": len(synced_photos)
            },
            "mapping": {
                "tickets": synced_tickets,
                "messages": synced_messages
            },
            "errors": errors
        }
    
    except Exception as e:
        print(f"❌ Erreur sync offline: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur synchronisation: {str(e)}")

@app.get("/api/knowledge/offline-cache")
async def get_knowledge_for_offline_cache(
    domain: Optional[str] = None,
    language: str = "fr",
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db)
):
    """
    🔑 ENDPOINT CACHE OFFLINE
    Télécharger la base de connaissances pour utilisation offline (RAG local)
    """
    query = db.query(KnowledgeItem).filter(KnowledgeItem.language == language)
    
    if domain:
        query = query.filter(KnowledgeItem.domain == domain)
    
    items = query.order_by(KnowledgeItem.created_at.desc()).limit(limit).all()
    
    return {
        "items": [
            {
                "id": item.id,
                "domain": item.domain,
                "title": item.title,
                "question": item.question,
                "answer": item.answer,
                "tags": json.loads(item.tags) if item.tags else [],
                "keywords": [tag.lower() for tag in (json.loads(item.tags) if item.tags else [])]
            }
            for item in items
        ],
        "total": len(items),
        "cached_at": datetime.utcnow().isoformat()
    }

# ==========================================
# ENDPOINTS FREEMIUM & DIALOGUE
# ==========================================

class SendMessageRequest(BaseModel):
    ticket_id: int
    sender_type: str  # 'user' ou 'expert'
    content: str

@app.get("/api/user-status")
async def get_user_status(phone: str, db: Session = Depends(get_db)):
    """Retourne le statut premium et les limites de messages d'un utilisateur"""
    user = db.query(User).filter(User.phone_number == phone).first()
    
    if not user:
        # Créer l'utilisateur s'il n'existe pas
        user = User(
            phone_number=phone,
            is_premium=False,
            messages_used=0,
            messages_limit=1
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Vérifier si l'abonnement premium est expiré
    if user.is_premium and user.premium_expires_at:
        if user.premium_expires_at < datetime.utcnow():
            user.is_premium = False
            user.messages_limit = 1
            db.commit()
    
    return {
        "is_premium": user.is_premium,
        "messages_used": user.messages_used,
        "messages_limit": user.messages_limit if user.is_premium else 1,
        "premium_expires_at": user.premium_expires_at.isoformat() if user.premium_expires_at else None
    }

@app.post("/api/send-message")
async def send_message(request: SendMessageRequest, db: Session = Depends(get_db)):
    """Envoie un message dans un ticket (user ou expert)"""
    
    # Récupérer le ticket
    ticket = db.query(Ticket).filter(Ticket.id == request.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket non trouvé")
    
    # Récupérer l'utilisateur
    user = db.query(User).filter(User.id == ticket.user_id).first()
    
    if request.sender_type == 'user':
        # Vérifier les limites de messages
        user_messages = db.query(Message).filter(
            Message.ticket_id == request.ticket_id,
            Message.sender_type == 'user'
        ).count()
        
        limit = user.messages_limit if user.is_premium else 1
        
        if user_messages >= limit:
            raise HTTPException(
                status_code=403, 
                detail=f"Limite de messages atteinte. Version gratuite : 1 message. Premium : 10 messages."
            )
    
    # Créer le message
    message = Message(
        ticket_id=request.ticket_id,
        sender_type=request.sender_type,
        sender_id=user.id if request.sender_type == 'user' else None,
        content=request.content,
        channel='web'
    )
    db.add(message)
    db.commit()
    
    return {"success": True, "message": "Message envoyé"}

# ==========================================
# LANCEMENT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("SONGRA - Backend avec IA complète")
    print("Version 5.0 - Analyse IA texte + photo")
    print("=" * 50)
    print("Serveur démarré sur http://localhost:8000")
    print("Test expert: test@resolvehub.bf / test123")
    print("=" * 50)
    
    # Créer l'expert test au démarrage
    try:
        db = SessionLocal()
        existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
        if not existing:
            expert = Expert(
                email="test@resolvehub.bf",
                password_hash=hash_password("test123"),
                full_name="Expert Test IA",
                specialization="agriculture",
                is_active=True
            )
            db.add(expert)
            db.commit()
            print("✓ Expert test créé: test@resolvehub.bf / test123")
        # Charger la base de connaissances locale depuis le JSON
        try:
            load_knowledge_from_json(db)
            total_items = db.query(KnowledgeItem).count()
            print(f"✓ Base de connaissances chargée ({total_items} fiches)")
        except Exception as e_load:
            print(f"⚠️ Erreur chargement base de connaissances: {e_load}")
        db.close()
    except Exception as e:
        print(f"⚠️ Erreur création expert: {e}")
    
    # Le mode reload est plutôt à utiliser avec la commande uvicorn en ligne
    # de commande (ex: `uvicorn main:app --reload`). Ici on garde un run simple.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)