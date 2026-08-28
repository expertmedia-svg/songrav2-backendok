"""Worker borne pour la file Voice Alert.

Le processus ne contient que MockProvider dans cette phase et ne peut donc
effectuer aucun appel téléphonique réel.
"""
import os
import time

from main import SessionLocal
from voice_alert import process_due_jobs


def run() -> None:
    batch_size = max(1, min(int(os.getenv("MOCK_WORKER_BATCH_SIZE", "25")), 100))
    poll_seconds = max(1, int(os.getenv("MOCK_WORKER_POLL_SECONDS", "5")))
    once = os.getenv("MOCK_WORKER_ONCE", "false").lower() == "true"
    while True:
        stats = process_due_jobs(SessionLocal, batch_size)
        print(f"[VOICE-ALERT-MOCK] {stats}", flush=True)
        if once:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    run()
