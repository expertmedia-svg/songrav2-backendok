from pathlib import Path
import sys
import asyncio
from datetime import datetime

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def _db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    main.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_classifies_general_farming_and_livestock_techniques():
    farming = main.ai_engine.classify(
        "Quand semer le maïs et comment améliorer la production ?"
    )
    livestock = main.ai_engine.classify(
        "Comment nourrir mes poules pour améliorer la ponte ?"
    )
    assert farming["category"] == "agriculture"
    assert farming["question_intent"] == "agricultural_technique"
    assert livestock["category"] == "elevage"
    assert livestock["question_intent"] == "livestock_technique"


def test_courses_are_scoped_to_the_users_organization():
    db = _db()
    own = main.AcademyCourseDB(
        title="Calendrier de semis du maïs", course_type="technique", crop="maïs",
        summary="Choisir la date et réussir le semis", status="published",
        organization_id=10,
    )
    other = main.AcademyCourseDB(
        title="Semis du maïs ONG voisine", course_type="technique", crop="maïs",
        summary="Cours réservé à une autre communauté", status="published",
        organization_id=20,
    )
    db.add_all([own, other])
    db.commit()
    user = main.User(phone_number="+22670000001", organization_id=10)
    visible = main._scope_courses_for_user(db.query(main.AcademyCourseDB), user).all()
    assert [course.id for course in visible] == [own.id]
    recommendation = main._find_academy_course_match(
        db, "quand faire le semis du maïs", domain="agriculture", organization_id=10
    )
    assert recommendation["id"] == own.id


def test_validated_fiche_is_used_before_external_knowledge(monkeypatch):
    db = _db()
    db.add(main.KnowledgeItem(
        domain="agriculture", title="Rouille du maïs",
        question="Des taches orange apparaissent sur les feuilles",
        answer="Retirer les feuilles atteintes et surveiller la parcelle.",
        tags='["maïs", "taches", "rouille"]', language="fr",
    ))
    db.commit()
    monkeypatch.setattr(main, "generate_llm_answer", lambda **kwargs: "Réponse fondée sur la fiche")
    result = main.resolve_knowledge_answer(
        db, "agriculture", "Mon maïs a une maladie avec des taches orange"
    )
    assert result["knowledge_mode"] == "rag_strict"
    assert result["llm_answer"] == "Réponse fondée sur la fiche"
    assert result["rag_items"][0]["title"] == "Rouille du maïs"


def test_calendar_question_uses_general_ai_and_rejects_generic_course(monkeypatch):
    db = _db()
    db.add(main.AcademyCourseDB(
        title="Installer la culture du maïs", course_type="culture", crop="maïs",
        summary="Préparer le champ et entretenir la culture", status="published",
        organization_id=None,
    ))
    db.commit()

    async def answer(**kwargs):
        return "Au Burkina Faso, semez après une pluie utile selon votre zone."

    monkeypatch.setattr(main.v2_services, "gemini_llm_general_knowledge", answer)
    result = main.resolve_knowledge_answer(
        db, "agriculture", "À quelle période semer le maïs ?"
    )
    assert result["knowledge_mode"] == "llm_general_knowledge"
    assert result["recommended_course"] is None
    assert "pluie utile" in result["llm_answer"]


def test_legacy_assistant_rejects_exhausted_analysis_quota():
    db = _db()
    user = main.User(phone_number="+22670000002")
    db.add(user)
    db.commit()
    db.refresh(user)
    db.add_all([
        main.UsageEventDB(user_id=user.id, resource="analyses", source="test", created_at=datetime.utcnow())
        for _ in range(3)
    ])
    db.commit()
    request = main.MessageCreate(
        phone_number=user.phone_number,
        content="Quand semer le maïs ?",
        category="agriculture",
    )
    with pytest.raises(HTTPException) as denied:
        asyncio.run(main.assistant_query(request, db))
    assert denied.value.status_code == 402
