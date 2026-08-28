from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from voice_alert import AlertBase, create_router


def build_client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    AlertBase.metadata.create_all(engine)
    user = SimpleNamespace(id=1, phone_number="+22670000001")
    expert = SimpleNamespace(id=10, email="expert@songra.test", role="admin")

    def get_db():
        db = sessions()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(create_router(get_db, lambda: user, lambda: expert, lambda: expert, sessions))
    return TestClient(app), sessions


def test_complete_mock_workflow_and_idempotence():
    client, _ = build_client()
    plot = {
        "id": "plot-1", "name": "Champ maïs", "crop_name": "maïs", "area_hectares": 2,
        "latitude": 12.25, "longitude": -1.55, "village": "Village Demo",
        "location_accuracy_m": 8, "location_source": "gps", "updated_at": "2026-08-28T10:00:00",
    }
    assert client.post("/api/voice-alert/plots/sync", json={"plots": [plot]}).status_code == 200
    assert client.put("/api/voice-alert/preferences", json={
        "agricultural_alerts": True, "voice_enabled": True, "sms_enabled": True,
        "preferred_language": "moore", "primary_phone": "+22670000001",
    }).status_code == 200

    reports = []
    for index in range(3):
        response = client.post("/api/voice-alert/reports", json={
            "threat_type": "chenille legionnaire", "crop_or_species": "maïs",
            "diagnosis_label": "attaque foliaire", "description": f"Signalement {index}",
            "latitude": 12.25 + index * 0.001, "longitude": -1.55, "confidence": 0.8,
        })
        assert response.status_code == 200
        reports.append(response.json())
    incident = reports[-1]["incident"]
    assert incident["report_count"] == 3
    assert len({item["incident"]["id"] for item in reports}) == 1

    assert client.post(f'/api/voice-alert/incidents/{incident["id"]}/validate', json={
        "decision": "VALIDATE", "severity": "HIGH", "radius_km": 15,
    }).status_code == 200
    estimate = client.get(f'/api/voice-alert/incidents/{incident["id"]}/recipients/estimate').json()
    assert estimate["count"] == 1

    campaign = client.post("/api/voice-alert/campaigns", json={
        "incident_id": incident["id"], "name": "Demo chenille",
        "messages": {"fr": "Alerte SONGRA", "moore": "Songra kibare"},
    }).json()
    campaign_id = campaign["id"]
    assert client.post(f"/api/voice-alert/campaigns/{campaign_id}/approve").json()["status"] == "APPROVED"
    first_queue = client.post(f"/api/voice-alert/campaigns/{campaign_id}/queue")
    assert first_queue.status_code == 200
    # Une seconde mise en file est refusee: aucun doublon de job/appel.
    assert client.post(f"/api/voice-alert/campaigns/{campaign_id}/queue").status_code == 409
    client.post("/api/voice-alert/dev/process-jobs?batch_size=25")
    campaign_state = client.get("/api/voice-alert/campaigns").json()["campaigns"][0]
    assert campaign_state["provider"] == "mock"
    assert campaign_state["real_calls"] is False
    assert campaign_state["status"] == "COMPLETED"


def test_consent_is_required_and_can_be_withdrawn():
    client, _ = build_client()
    client.post("/api/voice-alert/plots/sync", json={"plots": [{
        "id": "plot-2", "name": "Champ", "crop_name": "maïs", "area_hectares": 1,
        "latitude": 12.25, "longitude": -1.55,
    }]})
    report = client.post("/api/voice-alert/reports", json={
        "threat_type": "ravageur", "crop_or_species": "maïs", "description": "Feuilles attaquées",
        "latitude": 12.25, "longitude": -1.55,
    }).json()
    incident_id = report["incident"]["id"]
    assert client.get(f"/api/voice-alert/incidents/{incident_id}/recipients/estimate").json()["count"] == 0
    client.put("/api/voice-alert/preferences", json={
        "agricultural_alerts": True, "voice_enabled": True, "sms_enabled": False,
        "primary_phone": "+22670000001", "withdraw_consent": True,
    })
    assert client.get(f"/api/voice-alert/incidents/{incident_id}/recipients/estimate").json()["count"] == 0


def test_requested_demo_seeds_25_fictitious_maize_recipients():
    client, _ = build_client()
    demo = client.post("/api/voice-alert/dev/seed-demo")
    assert demo.status_code == 200
    assert demo.json()["recipient_count"] == 25
    assert demo.json()["real_calls"] is False
    campaign_id = demo.json()["campaign_id"]
    queued = client.post(f"/api/voice-alert/campaigns/{campaign_id}/queue")
    assert queued.status_code == 200
    assert queued.json()["provider"] == "mock"
    client.post("/api/voice-alert/dev/process-jobs?batch_size=100")
    state = client.get("/api/voice-alert/campaigns").json()["campaigns"][0]
    assert state["status"] == "COMPLETED"
    assert sum(state["results"].values()) == 25
