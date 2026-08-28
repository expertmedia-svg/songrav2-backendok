from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import voice_alert


def test_expert_password_bcrypt_and_legacy_verification():
    modern = main.hash_expert_password("MotDePasseSolide42")
    assert modern.startswith("$2")
    assert main.verify_password("MotDePasseSolide42", modern)
    assert not main.verify_password("incorrect", modern)
    assert main.verify_password("legacy", main.hash_password("legacy"))


def test_expert_jwt_is_signed_expiring_and_rbac(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine)
    main.Base.metadata.create_all(engine)
    voice_alert.AlertBase.metadata.create_all(engine)
    db = sessions()
    expert = main.Expert(
        email="expert-security@songra.test", password_hash=main.hash_expert_password("secret"),
        full_name="Expert Sécurité", role="expert", is_active=True,
    )
    db.add(expert)
    db.commit()
    db.refresh(expert)
    monkeypatch.setattr(main, "SessionLocal", sessions)
    token = main.create_expert_access_token(expert)
    authenticated = main.get_current_expert(f"Bearer {token}", db)
    assert authenticated.id == expert.id
    with pytest.raises(HTTPException) as denied:
        main.get_current_admin_expert(authenticated)
    assert denied.value.status_code == 403
    expert.role = "admin"
    assert main.get_current_admin_expert(expert).id == expert.id
    with pytest.raises(HTTPException):
        main.get_current_expert("Bearer token_1_unsigned", db)
    db.close()
