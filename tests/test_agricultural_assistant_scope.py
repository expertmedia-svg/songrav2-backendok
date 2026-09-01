from pathlib import Path
import sys

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
        title="Semis du maïs", course_type="technique", crop="maïs",
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
        domain="agriculture", title="Semis du maïs",
        question="Quand semer le maïs ?",
        answer="Semer après une pluie utile quand le sol est bien humide.",
        tags='["maïs", "semis", "pluie"]', language="fr",
    ))
    db.commit()
    monkeypatch.setattr(main, "generate_llm_answer", lambda **kwargs: "Réponse fondée sur la fiche")
    result = main.resolve_knowledge_answer(
        db, "agriculture", "Quand semer le maïs après la pluie ?"
    )
    assert result["knowledge_mode"] == "rag_strict"
    assert result["llm_answer"] == "Réponse fondée sur la fiche"
    assert result["rag_items"][0]["title"] == "Semis du maïs"

