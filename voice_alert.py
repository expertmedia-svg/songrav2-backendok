"""Fondations SONGRA Voice Alert, sans integration Voice operateur reelle.

Ce module isole le domaine agricole du domaine SOS historique. Il utilise une
file persistante en base et un fournisseur simule deterministe afin que le
workflow complet puisse etre demontre sans effectuer le moindre appel reel.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
import os
import secrets
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, and_, or_
from sqlalchemy.orm import Session, declarative_base


AlertBase = declarative_base()


class FarmPlotDB(AlertBase):
    __tablename__ = "farm_plots"
    id = Column(String, primary_key=True)
    owner_user_id = Column(Integer, nullable=False, index=True)
    name = Column(String, nullable=False)
    crop_name = Column(String, nullable=False, index=True)
    area_hectares = Column(Float, nullable=False, default=0)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    geometry_json = Column(Text, nullable=True)
    village = Column(String, nullable=True, index=True)
    commune = Column(String, nullable=True, index=True)
    province = Column(String, nullable=True, index=True)
    region = Column(String, nullable=True, index=True)
    location_accuracy_m = Column(Float, nullable=True)
    location_source = Column(String, nullable=False, default="manual")
    client_updated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AlertPreferenceDB(AlertBase):
    __tablename__ = "alert_preferences"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, nullable=False, index=True)
    agricultural_alerts = Column(Boolean, nullable=False, default=False)
    voice_enabled = Column(Boolean, nullable=False, default=False)
    sms_enabled = Column(Boolean, nullable=False, default=False)
    preferred_language = Column(String, nullable=False, default="fr")
    primary_phone = Column(String, nullable=True)
    alternate_phone = Column(String, nullable=True)
    consented_at = Column(DateTime, nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)
    consent_version = Column(String, nullable=False, default="voice-alert-v1")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgriculturalIncidentDB(AlertBase):
    __tablename__ = "agricultural_incidents"
    id = Column(Integer, primary_key=True)
    incident_code = Column(String, unique=True, nullable=False, index=True)
    threat_type = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, default="agriculture")
    crop_or_species = Column(String, nullable=False, index=True)
    diagnosis_label = Column(String, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String, nullable=False, default="MEDIUM")
    status = Column(String, nullable=False, default="PROPOSED", index=True)
    center_latitude = Column(Float, nullable=True, index=True)
    center_longitude = Column(Float, nullable=True, index=True)
    radius_km = Column(Float, nullable=False, default=15)
    report_count = Column(Integer, nullable=False, default=1)
    confidence = Column(Float, nullable=False, default=0)
    validated_by = Column(Integer, nullable=True)
    validated_at = Column(DateTime, nullable=True)
    rejected_by = Column(Integer, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class IncidentReportDB(AlertBase):
    __tablename__ = "incident_reports"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, nullable=False, index=True)
    reporter_user_id = Column(Integer, nullable=True, index=True)
    source_type = Column(String, nullable=False, default="manual")
    source_id = Column(String, nullable=True)
    threat_type = Column(String, nullable=False, index=True)
    crop_or_species = Column(String, nullable=False, index=True)
    diagnosis_label = Column(String, nullable=True)
    description = Column(Text, nullable=False)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=0.5)
    evidence_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class IncidentValidationDB(AlertBase):
    __tablename__ = "incident_validations"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, nullable=False, index=True)
    actor_expert_id = Column(Integer, nullable=False, index=True)
    decision = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class VoiceCampaignDB(AlertBase):
    __tablename__ = "voice_campaigns"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, nullable=False, index=True)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="DRAFT", index=True)
    priority = Column(Integer, nullable=False, default=5)
    channel = Column(String, nullable=False, default="VOICE_MOCK")
    radius_km = Column(Float, nullable=False, default=15)
    message_by_language_json = Column(Text, nullable=False, default="{}")
    recipient_count = Column(Integer, nullable=False, default=0)
    estimated_budget = Column(Float, nullable=False, default=0)
    max_recipients = Column(Integer, nullable=False, default=500)
    max_retries = Column(Integer, nullable=False, default=2)
    requires_second_approval = Column(Boolean, nullable=False, default=False)
    first_approved_by = Column(Integer, nullable=True)
    second_approved_by = Column(Integer, nullable=True)
    created_by = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class CampaignRecipientDB(AlertBase):
    __tablename__ = "campaign_recipients"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    farm_plot_id = Column(String, nullable=False, index=True)
    phone_number = Column(String, nullable=False)
    language = Column(String, nullable=False, default="fr")
    status = Column(String, nullable=False, default="PENDING", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    provider_call_id = Column(String, nullable=True, unique=True)
    acknowledged_at = Column(DateTime, nullable=True)
    last_attempt_at = Column(DateTime, nullable=True)
    result_json = Column(Text, nullable=True)


class TelecomJobDB(AlertBase):
    __tablename__ = "telecom_jobs"
    id = Column(Integer, primary_key=True)
    idempotency_key = Column(String, unique=True, nullable=False)
    campaign_id = Column(Integer, nullable=False, index=True)
    recipient_id = Column(Integer, nullable=False, index=True)
    status = Column(String, nullable=False, default="QUEUED", index=True)
    priority = Column(Integer, nullable=False, default=5, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    available_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    locked_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class AlertAuditLogDB(AlertBase):
    __tablename__ = "alert_audit_logs"
    id = Column(Integer, primary_key=True)
    actor_type = Column(String, nullable=False)
    actor_id = Column(Integer, nullable=True)
    action = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=False, index=True)
    entity_id = Column(String, nullable=False, index=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class RevokedExpertTokenDB(AlertBase):
    __tablename__ = "revoked_expert_tokens"
    jti_hash = Column(String, primary_key=True)
    expert_id = Column(Integer, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked_at = Column(DateTime, default=datetime.utcnow)


class PlotIn(BaseModel):
    id: str
    name: str
    crop_name: str
    area_hectares: float = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None
    village: Optional[str] = None
    commune: Optional[str] = None
    province: Optional[str] = None
    region: Optional[str] = None
    location_accuracy_m: Optional[float] = None
    location_source: str = "manual"
    updated_at: Optional[datetime] = None


class PlotSyncIn(BaseModel):
    plots: List[PlotIn]


class PreferenceIn(BaseModel):
    agricultural_alerts: bool
    voice_enabled: bool
    sms_enabled: bool
    preferred_language: str = "fr"
    primary_phone: Optional[str] = None
    alternate_phone: Optional[str] = None
    withdraw_consent: bool = False


class ReportIn(BaseModel):
    threat_type: str
    crop_or_species: str
    diagnosis_label: Optional[str] = None
    description: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = Field(0.5, ge=0, le=1)
    source_type: str = "manual"
    source_id: Optional[str] = None
    evidence: Dict[str, Any] = {}


class ValidationIn(BaseModel):
    decision: str
    notes: Optional[str] = None
    severity: Optional[str] = None
    radius_km: Optional[float] = Field(None, gt=0, le=500)


class CampaignIn(BaseModel):
    incident_id: int
    name: str
    messages: Dict[str, str]
    priority: int = Field(5, ge=1, le=10)
    max_recipients: int = Field(500, ge=1, le=100000)
    max_retries: int = Field(2, ge=0, le=10)


def _distance_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = math.radians(b_lat - a_lat), math.radians(b_lng - a_lng)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _audit(db: Session, actor: Any, action: str, entity: str, entity_id: Any, details: Any = None) -> None:
    db.add(AlertAuditLogDB(
        actor_type="expert" if hasattr(actor, "email") else "user",
        actor_id=getattr(actor, "id", None), action=action, entity_type=entity,
        entity_id=str(entity_id), details_json=json.dumps(details or {}, ensure_ascii=False),
    ))


def _serialize_incident(item: AgriculturalIncidentDB, db: Session) -> Dict[str, Any]:
    return {
        "id": item.id, "incident_code": item.incident_code, "threat_type": item.threat_type,
        "category": item.category, "crop_or_species": item.crop_or_species,
        "diagnosis_label": item.diagnosis_label, "title": item.title,
        "description": item.description, "severity": item.severity, "status": item.status,
        "latitude": item.center_latitude, "longitude": item.center_longitude,
        "radius_km": item.radius_km, "report_count": item.report_count,
        "confidence": item.confidence,
        "reports": [{"id": r.id, "description": r.description, "latitude": r.latitude,
                     "longitude": r.longitude, "confidence": r.confidence, "created_at": r.created_at.isoformat()}
                    for r in db.query(IncidentReportDB).filter(IncidentReportDB.incident_id == item.id).all()],
        "created_at": item.created_at.isoformat(),
    }


def _resolve_recipients(db: Session, incident: AgriculturalIncidentDB, limit: int) -> List[Dict[str, Any]]:
    crop = incident.crop_or_species.strip().lower()
    plots = db.query(FarmPlotDB).filter(FarmPlotDB.crop_name.ilike(crop)).all()
    resolved: Dict[int, Dict[str, Any]] = {}
    for plot in plots:
        if plot.latitude is None or plot.longitude is None or incident.center_latitude is None or incident.center_longitude is None:
            continue
        distance = _distance_km(incident.center_latitude, incident.center_longitude, plot.latitude, plot.longitude)
        if distance > incident.radius_km:
            continue
        pref = db.query(AlertPreferenceDB).filter(AlertPreferenceDB.user_id == plot.owner_user_id).first()
        if not pref or not pref.agricultural_alerts or not pref.voice_enabled or pref.withdrawn_at:
            continue
        phone = pref.primary_phone or pref.alternate_phone
        if not phone:
            continue
        resolved.setdefault(plot.owner_user_id, {
            "user_id": plot.owner_user_id, "farm_plot_id": plot.id, "phone_number": phone,
            "language": pref.preferred_language, "distance_km": round(distance, 2),
            "reason": f"Parcelle {plot.name}, culture {plot.crop_name}, rayon {incident.radius_km} km",
        })
        if len(resolved) >= limit:
            break
    return list(resolved.values())


class MockProvider:
    """Fournisseur strictement local : aucun acces reseau."""
    name = "mock"

    def capabilities(self) -> Dict[str, Any]:
        return {"voice": True, "sms": True, "callbacks": True, "dtmf": "SIMULATED_ONLY", "real_calls": False}

    def start_voice_call(self, recipient: CampaignRecipientDB) -> Dict[str, Any]:
        bucket = recipient.id % 10
        final = "NO_ANSWER" if bucket in {1, 2} else "BUSY" if bucket == 3 else "FAILED" if bucket == 4 else "COMPLETED"
        acknowledged = final == "COMPLETED" and bucket % 2 == 0
        return {
            "provider": self.name, "provider_call_id": f"mock-{recipient.campaign_id}-{recipient.id}",
            "events": ["QUEUED", "CALLING", "RINGING"] + (["ANSWERED", "COMPLETED"] if final == "COMPLETED" else [final]),
            "final_status": final, "acknowledged": acknowledged,
            "dtmf": "1" if acknowledged else None, "simulation": True,
        }


def process_due_jobs(session_factory: Callable[[], Session], batch_size: int = 25) -> Dict[str, int]:
    """Traite un lot borne. La file DB reste la source de verite durable."""
    db = session_factory()
    provider = MockProvider()
    stats = {"processed": 0, "completed": 0, "retried": 0, "dlq": 0}
    try:
        jobs = db.query(TelecomJobDB).filter(
            TelecomJobDB.status == "QUEUED", TelecomJobDB.available_at <= datetime.utcnow()
        ).order_by(TelecomJobDB.priority.desc(), TelecomJobDB.id.asc()).limit(max(1, min(batch_size, 100))).all()
        for job in jobs:
            campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.id == job.campaign_id).first()
            recipient = db.query(CampaignRecipientDB).filter(CampaignRecipientDB.id == job.recipient_id).first()
            if not campaign or not recipient or campaign.status in {"PAUSED", "CANCELLED"}:
                continue
            job.status, job.locked_at = "PROCESSING", datetime.utcnow()
            campaign.status = "RUNNING"
            recipient.status = "CALLING"
            db.commit()
            try:
                result = provider.start_voice_call(recipient)
                recipient.provider_call_id = result["provider_call_id"]
                recipient.status = "ACKNOWLEDGED" if result["acknowledged"] else result["final_status"]
                recipient.attempts += 1
                recipient.last_attempt_at = datetime.utcnow()
                recipient.result_json = json.dumps(result)
                if result["acknowledged"]:
                    recipient.acknowledged_at = datetime.utcnow()
                job.status, job.completed_at = "COMPLETED", datetime.utcnow()
                stats["completed"] += 1
            except Exception as exc:  # defensive: futur provider
                job.attempts += 1
                job.last_error = str(exc)[:1000]
                if job.attempts >= job.max_attempts:
                    job.status = "DLQ"
                    recipient.status = "FAILED"
                    stats["dlq"] += 1
                else:
                    job.status = "QUEUED"
                    job.available_at = datetime.utcnow() + timedelta(seconds=min(3600, 2 ** job.attempts * 10))
                    stats["retried"] += 1
            stats["processed"] += 1
            db.commit()
        campaign_ids = {job.campaign_id for job in jobs}
        for campaign_id in campaign_ids:
            remaining = db.query(TelecomJobDB).filter(TelecomJobDB.campaign_id == campaign_id, TelecomJobDB.status.in_(["QUEUED", "PROCESSING"])).count()
            if remaining == 0:
                campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.id == campaign_id).first()
                if campaign and campaign.status not in {"PAUSED", "CANCELLED"}:
                    campaign.status, campaign.completed_at = "COMPLETED", datetime.utcnow()
        db.commit()
        return stats
    finally:
        db.close()


def is_expert_token_revoked(session_factory: Callable[[], Session], jti: str) -> bool:
    db = session_factory()
    try:
        return db.query(RevokedExpertTokenDB).filter(RevokedExpertTokenDB.jti_hash == sha256(jti.encode()).hexdigest()).first() is not None
    finally:
        db.close()


def revoke_expert_token(session_factory: Callable[[], Session], expert_id: int, jti: str, expires_at: datetime) -> None:
    db = session_factory()
    try:
        key = sha256(jti.encode()).hexdigest()
        if not db.query(RevokedExpertTokenDB).filter(RevokedExpertTokenDB.jti_hash == key).first():
            db.add(RevokedExpertTokenDB(jti_hash=key, expert_id=expert_id, expires_at=expires_at))
            db.commit()
    finally:
        db.close()


def create_router(get_db: Callable, get_user: Callable, get_expert: Callable, get_admin: Callable, session_factory: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/voice-alert", tags=["Voice Alert - simulation"])

    @router.get("/capabilities")
    def capabilities(_: Any = Depends(get_expert)):
        return MockProvider().capabilities()

    @router.post("/plots/sync")
    def sync_plots(payload: PlotSyncIn, user: Any = Depends(get_user), db: Session = Depends(get_db)):
        returned = []
        for data in payload.plots:
            item = db.query(FarmPlotDB).filter(FarmPlotDB.id == data.id, FarmPlotDB.owner_user_id == user.id).first()
            if not item:
                item = FarmPlotDB(id=data.id, owner_user_id=user.id)
                db.add(item)
            for key, value in data.model_dump().items():
                if key == "geometry":
                    item.geometry_json = json.dumps(value) if value else None
                elif key == "updated_at":
                    item.client_updated_at = value
                else:
                    setattr(item, key, value)
            returned.append(data.id)
        _audit(db, user, "plots.sync", "farm_plot", ",".join(returned), {"count": len(returned)})
        db.commit()
        return {"synced": returned, "count": len(returned)}

    @router.get("/plots")
    def list_plots(user: Any = Depends(get_user), db: Session = Depends(get_db)):
        items = db.query(FarmPlotDB).filter(FarmPlotDB.owner_user_id == user.id).all()
        return {"plots": [{"id": p.id, "name": p.name, "crop_name": p.crop_name, "area_hectares": p.area_hectares,
                            "latitude": p.latitude, "longitude": p.longitude, "village": p.village,
                            "commune": p.commune, "province": p.province, "region": p.region,
                            "location_accuracy_m": p.location_accuracy_m, "location_source": p.location_source}
                           for p in items]}

    @router.put("/preferences")
    def update_preferences(payload: PreferenceIn, user: Any = Depends(get_user), db: Session = Depends(get_db)):
        pref = db.query(AlertPreferenceDB).filter(AlertPreferenceDB.user_id == user.id).first() or AlertPreferenceDB(user_id=user.id)
        if pref.id is None:
            db.add(pref)
        for key, value in payload.model_dump(exclude={"withdraw_consent"}).items():
            setattr(pref, key, value)
        if payload.withdraw_consent:
            pref.withdrawn_at = datetime.utcnow()
            pref.voice_enabled = pref.sms_enabled = pref.agricultural_alerts = False
        else:
            pref.withdrawn_at = None
            pref.consented_at = datetime.utcnow()
        _audit(db, user, "preferences.update", "alert_preference", user.id)
        db.commit()
        return {"saved": True, "consented": pref.withdrawn_at is None and pref.agricultural_alerts}

    @router.post("/reports")
    def create_report(payload: ReportIn, user: Any = Depends(get_user), db: Session = Depends(get_db)):
        since = datetime.utcnow() - timedelta(days=int(os.getenv("ALERT_CLUSTER_WINDOW_DAYS", "7")))
        candidates = db.query(AgriculturalIncidentDB).filter(
            AgriculturalIncidentDB.status.in_(["PROPOSED", "VALIDATED"]),
            AgriculturalIncidentDB.threat_type.ilike(payload.threat_type.strip()),
            AgriculturalIncidentDB.crop_or_species.ilike(payload.crop_or_species.strip()),
            AgriculturalIncidentDB.created_at >= since,
        ).all()
        incident = next((item for item in candidates if payload.latitude is not None and payload.longitude is not None
                         and item.center_latitude is not None and item.center_longitude is not None
                         and _distance_km(payload.latitude, payload.longitude, item.center_latitude, item.center_longitude)
                         <= float(os.getenv("ALERT_CLUSTER_RADIUS_KM", "20"))), None)
        if not incident:
            incident = AgriculturalIncidentDB(
                incident_code=f"AG-{datetime.utcnow().year}-{secrets.token_hex(3).upper()}",
                threat_type=payload.threat_type.strip(), crop_or_species=payload.crop_or_species.strip(),
                diagnosis_label=payload.diagnosis_label, title=f"{payload.threat_type} - {payload.crop_or_species}",
                description=payload.description, center_latitude=payload.latitude,
                center_longitude=payload.longitude, confidence=payload.confidence,
            )
            db.add(incident)
            db.flush()
        else:
            old_count = incident.report_count
            incident.report_count += 1
            incident.confidence = round((incident.confidence * old_count + payload.confidence) / incident.report_count, 3)
        report = IncidentReportDB(
            incident_id=incident.id, reporter_user_id=user.id, source_type=payload.source_type,
            source_id=payload.source_id, threat_type=payload.threat_type,
            crop_or_species=payload.crop_or_species, diagnosis_label=payload.diagnosis_label,
            description=payload.description, latitude=payload.latitude, longitude=payload.longitude,
            confidence=payload.confidence, evidence_json=json.dumps(payload.evidence, ensure_ascii=False),
        )
        db.add(report)
        _audit(db, user, "report.create", "agricultural_incident", incident.id, {"clustered": incident.report_count > 1})
        db.commit()
        return {"incident": _serialize_incident(incident, db), "report_id": report.id, "clustered": incident.report_count > 1}

    @router.get("/incidents")
    def list_incidents(status: Optional[str] = None, _: Any = Depends(get_expert), db: Session = Depends(get_db)):
        query = db.query(AgriculturalIncidentDB)
        if status:
            query = query.filter(AgriculturalIncidentDB.status == status.upper())
        return {"incidents": [_serialize_incident(i, db) for i in query.order_by(AgriculturalIncidentDB.created_at.desc()).limit(200).all()]}

    @router.post("/incidents/{incident_id}/validate")
    def validate_incident(incident_id: int, payload: ValidationIn, expert: Any = Depends(get_expert), db: Session = Depends(get_db)):
        incident = db.query(AgriculturalIncidentDB).filter(AgriculturalIncidentDB.id == incident_id).first()
        if not incident:
            raise HTTPException(404, "Incident introuvable")
        decision = payload.decision.upper()
        if decision not in {"VALIDATE", "REJECT"}:
            raise HTTPException(422, "Decision attendue: VALIDATE ou REJECT")
        if decision == "VALIDATE":
            incident.status, incident.validated_by, incident.validated_at = "VALIDATED", expert.id, datetime.utcnow()
            if payload.severity:
                incident.severity = payload.severity.upper()
            if payload.radius_km:
                incident.radius_km = payload.radius_km
        else:
            incident.status, incident.rejected_by, incident.rejected_at = "REJECTED", expert.id, datetime.utcnow()
        db.add(IncidentValidationDB(incident_id=incident.id, actor_expert_id=expert.id, decision=decision, notes=payload.notes))
        _audit(db, expert, f"incident.{decision.lower()}", "agricultural_incident", incident.id)
        db.commit()
        return _serialize_incident(incident, db)

    @router.get("/incidents/{incident_id}/recipients/estimate")
    def estimate(incident_id: int, expert: Any = Depends(get_expert), db: Session = Depends(get_db)):
        incident = db.query(AgriculturalIncidentDB).filter(AgriculturalIncidentDB.id == incident_id).first()
        if not incident:
            raise HTTPException(404, "Incident introuvable")
        recipients = _resolve_recipients(db, incident, 100000)
        return {"count": len(recipients), "recipients": recipients, "snapshot": False}

    @router.post("/campaigns")
    def create_campaign(payload: CampaignIn, expert: Any = Depends(get_expert), db: Session = Depends(get_db)):
        incident = db.query(AgriculturalIncidentDB).filter(AgriculturalIncidentDB.id == payload.incident_id).first()
        if not incident or incident.status != "VALIDATED":
            raise HTTPException(409, "L'incident doit etre valide avant la campagne")
        resolved = _resolve_recipients(db, incident, payload.max_recipients)
        second_threshold = int(os.getenv("CAMPAIGN_SECOND_APPROVAL_RECIPIENTS", "500"))
        estimated_budget = len(resolved) * float(os.getenv("MOCK_VOICE_UNIT_COST", "0"))
        budget_threshold = float(os.getenv("CAMPAIGN_SECOND_APPROVAL_BUDGET", "50000"))
        radius_threshold = float(os.getenv("CAMPAIGN_SECOND_APPROVAL_RADIUS_KM", "50"))
        requires_second = (
            len(resolved) >= second_threshold or incident.severity == "CRITICAL"
            or incident.radius_km >= radius_threshold or estimated_budget >= budget_threshold
        )
        campaign = VoiceCampaignDB(
            incident_id=incident.id, name=payload.name, status="PENDING_APPROVAL",
            priority=payload.priority, message_by_language_json=json.dumps(payload.messages, ensure_ascii=False),
            recipient_count=len(resolved), max_recipients=payload.max_recipients,
            max_retries=payload.max_retries, estimated_budget=estimated_budget,
            requires_second_approval=requires_second, created_by=expert.id,
        )
        db.add(campaign)
        db.flush()
        for item in resolved:
            db.add(CampaignRecipientDB(campaign_id=campaign.id, **{k: item[k] for k in ("user_id", "farm_plot_id", "phone_number", "language")}))
        _audit(db, expert, "campaign.create", "voice_campaign", campaign.id, {"recipients": len(resolved), "provider": "mock"})
        db.commit()
        return {"id": campaign.id, "status": campaign.status, "recipient_count": len(resolved), "requires_second_approval": requires_second}

    @router.post("/campaigns/{campaign_id}/approve")
    def approve_campaign(campaign_id: int, expert: Any = Depends(get_expert), db: Session = Depends(get_db)):
        campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.id == campaign_id).first()
        if not campaign or campaign.status not in {"PENDING_APPROVAL", "APPROVED"}:
            raise HTTPException(409, "Campagne non approuvable")
        if campaign.first_approved_by is None:
            campaign.first_approved_by = expert.id
        elif campaign.requires_second_approval and campaign.first_approved_by == expert.id:
            raise HTTPException(409, "La seconde validation doit venir d'un autre compte")
        else:
            campaign.second_approved_by = expert.id
        if not campaign.requires_second_approval or campaign.second_approved_by:
            campaign.status = "APPROVED"
        _audit(db, expert, "campaign.approve", "voice_campaign", campaign.id)
        db.commit()
        return {"id": campaign.id, "status": campaign.status, "requires_second_approval": campaign.requires_second_approval}

    @router.post("/campaigns/{campaign_id}/queue")
    def queue_campaign(campaign_id: int, tasks: BackgroundTasks, expert: Any = Depends(get_admin), db: Session = Depends(get_db)):
        campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.id == campaign_id).first()
        if not campaign or campaign.status != "APPROVED":
            raise HTTPException(409, "Campagne non approuvee")
        recipients = db.query(CampaignRecipientDB).filter(CampaignRecipientDB.campaign_id == campaign.id).all()
        for recipient in recipients:
            key = f"campaign:{campaign.id}:recipient:{recipient.id}:attempt:initial"
            if not db.query(TelecomJobDB).filter(TelecomJobDB.idempotency_key == key).first():
                db.add(TelecomJobDB(idempotency_key=key, campaign_id=campaign.id, recipient_id=recipient.id,
                                    priority=campaign.priority, max_attempts=campaign.max_retries + 1))
                recipient.status = "QUEUED"
        campaign.status = "QUEUED"
        _audit(db, expert, "campaign.queue", "voice_campaign", campaign.id, {"jobs": len(recipients)})
        db.commit()
        tasks.add_task(process_due_jobs, session_factory, min(25, int(os.getenv("MOCK_WORKER_BATCH_SIZE", "25"))))
        return {"id": campaign.id, "status": "QUEUED", "jobs": len(recipients), "provider": "mock", "real_calls": False}

    @router.post("/campaigns/{campaign_id}/{action}")
    def control_campaign(campaign_id: int, action: str, expert: Any = Depends(get_admin), db: Session = Depends(get_db)):
        campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.id == campaign_id).first()
        if not campaign:
            raise HTTPException(404, "Campagne introuvable")
        action = action.lower()
        mapping = {"pause": "PAUSED", "resume": "QUEUED", "cancel": "CANCELLED"}
        if action not in mapping:
            raise HTTPException(422, "Action attendue: pause, resume ou cancel")
        campaign.status = mapping[action]
        _audit(db, expert, f"campaign.{action}", "voice_campaign", campaign.id)
        db.commit()
        return {"id": campaign.id, "status": campaign.status}

    @router.get("/campaigns")
    def list_campaigns(_: Any = Depends(get_expert), db: Session = Depends(get_db)):
        result = []
        for c in db.query(VoiceCampaignDB).order_by(VoiceCampaignDB.created_at.desc()).limit(100).all():
            counts: Dict[str, int] = {}
            for r in db.query(CampaignRecipientDB).filter(CampaignRecipientDB.campaign_id == c.id).all():
                counts[r.status] = counts.get(r.status, 0) + 1
            result.append({"id": c.id, "incident_id": c.incident_id, "name": c.name, "status": c.status,
                           "recipient_count": c.recipient_count, "requires_second_approval": c.requires_second_approval,
                           "results": counts, "provider": "mock", "real_calls": False})
        return {"campaigns": result}

    @router.post("/dev/seed-demo")
    def seed_demo(expert: Any = Depends(get_admin), db: Session = Depends(get_db)):
        """Scenario reproductible demande, uniquement hors production."""
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise HTTPException(404, "Indisponible")
        code = "AG-DEMO-MAIS"
        incident = db.query(AgriculturalIncidentDB).filter(AgriculturalIncidentDB.incident_code == code).first()
        if not incident:
            incident = AgriculturalIncidentDB(
                incident_code=code, threat_type="chenille legionnaire", crop_or_species="maïs",
                diagnosis_label="attaque foliaire", title="Chenille légionnaire sur maïs",
                description="Trois producteurs signalent le même ravageur dans la zone de démonstration.",
                severity="HIGH", status="VALIDATED", center_latitude=12.25, center_longitude=-1.55,
                radius_km=15, report_count=3, confidence=0.87,
                validated_by=expert.id, validated_at=datetime.utcnow(),
            )
            db.add(incident)
            db.flush()
            for index in range(3):
                db.add(IncidentReportDB(
                    incident_id=incident.id, reporter_user_id=9000 + index, source_type="demo",
                    source_id=f"demo-report-{index + 1}", threat_type=incident.threat_type,
                    crop_or_species="maïs", diagnosis_label=incident.diagnosis_label,
                    description=f"Signalement terrain fictif {index + 1}",
                    latitude=12.25 + index * 0.003, longitude=-1.55 + index * 0.002, confidence=0.85,
                ))
        for index in range(25):
            user_id = 10000 + index
            plot_id = f"demo-maize-{index + 1:02d}"
            if not db.query(FarmPlotDB).filter(FarmPlotDB.id == plot_id).first():
                db.add(FarmPlotDB(
                    id=plot_id, owner_user_id=user_id, name=f"Parcelle maïs {index + 1}",
                    crop_name="maïs", area_hectares=1 + index / 10,
                    latitude=12.25 + (index % 5) * 0.005, longitude=-1.55 + (index // 5) * 0.005,
                    village="Village Démo", commune="Commune Démo", province="Province Démo",
                    region="Centre", location_accuracy_m=10, location_source="demo",
                ))
            if not db.query(AlertPreferenceDB).filter(AlertPreferenceDB.user_id == user_id).first():
                db.add(AlertPreferenceDB(
                    user_id=user_id, agricultural_alerts=True, voice_enabled=True, sms_enabled=True,
                    preferred_language=("fr", "moore", "dioula", "fulfulde")[index % 4],
                    primary_phone=f"+2267999{index:04d}", consented_at=datetime.utcnow(),
                ))
        db.flush()
        campaign = db.query(VoiceCampaignDB).filter(VoiceCampaignDB.name == "DÉMO — Chenille maïs 15 km").first()
        if not campaign:
            recipients = _resolve_recipients(db, incident, 25)
            campaign = VoiceCampaignDB(
                incident_id=incident.id, name="DÉMO — Chenille maïs 15 km", status="APPROVED",
                priority=8, message_by_language_json=json.dumps({
                    "fr": "Alerte SONGRA. Présence de chenilles dans votre zone. Vérifiez votre parcelle de maïs."
                }, ensure_ascii=False), recipient_count=len(recipients), max_recipients=25,
                max_retries=2, created_by=expert.id, first_approved_by=expert.id,
            )
            db.add(campaign)
            db.flush()
            for item in recipients:
                db.add(CampaignRecipientDB(campaign_id=campaign.id, **{k: item[k] for k in ("user_id", "farm_plot_id", "phone_number", "language")}))
        _audit(db, expert, "demo.seed", "voice_campaign", campaign.id, {"fictitious_recipients": 25})
        db.commit()
        return {"incident_id": incident.id, "campaign_id": campaign.id, "recipient_count": campaign.recipient_count,
                "next": [f"POST /api/voice-alert/campaigns/{campaign.id}/queue", "POST /api/voice-alert/dev/process-jobs"],
                "provider": "mock", "real_calls": False}

    @router.post("/dev/process-jobs")
    def process_jobs(batch_size: int = Query(25, ge=1, le=100), _: Any = Depends(get_admin)):
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise HTTPException(404, "Indisponible")
        return process_due_jobs(session_factory, batch_size)

    @router.get("/audit")
    def audit_logs(_: Any = Depends(get_admin), db: Session = Depends(get_db)):
        logs = db.query(AlertAuditLogDB).order_by(AlertAuditLogDB.created_at.desc()).limit(500).all()
        return {"logs": [{"id": x.id, "action": x.action, "entity_type": x.entity_type,
                           "entity_id": x.entity_id, "actor_id": x.actor_id,
                           "created_at": x.created_at.isoformat()} for x in logs]}

    return router
