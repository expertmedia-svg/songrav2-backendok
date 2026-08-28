"""Execute la demonstration locale Voice Alert sans authentification externe.

Le script surcharge uniquement les dependances d'identite dans le TestClient.
MockProvider reste le seul fournisseur charge et aucun acces Voice reseau n'est
possible.
"""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main


def run() -> None:
    admin = SimpleNamespace(id=1, email="demo-admin@local", role="admin")
    main.app.dependency_overrides[main.get_current_expert] = lambda: admin
    main.app.dependency_overrides[main.get_current_admin_expert] = lambda: admin
    try:
        with TestClient(main.app) as client:
            seeded = client.post("/api/voice-alert/dev/seed-demo")
            seeded.raise_for_status()
            demo = seeded.json()
            queued = client.post(f"/api/voice-alert/campaigns/{demo['campaign_id']}/queue")
            if queued.status_code not in {200, 409}:
                queued.raise_for_status()
            while True:
                processed = client.post("/api/voice-alert/dev/process-jobs?batch_size=100")
                processed.raise_for_status()
                if processed.json()["processed"] == 0:
                    break
            campaigns = client.get("/api/voice-alert/campaigns")
            campaigns.raise_for_status()
            selected = next(item for item in campaigns.json()["campaigns"] if item["id"] == demo["campaign_id"])
            print(json.dumps({"seed": demo, "campaign": selected, "safety": {
                "provider": "mock", "real_calls": False, "orange_voice_loaded": False,
            }}, ensure_ascii=False, indent=2))
    finally:
        main.app.dependency_overrides.clear()


if __name__ == "__main__":
    run()
