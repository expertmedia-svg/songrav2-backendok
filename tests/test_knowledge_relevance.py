import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import knowledge_relevance as relevance
import v2_services as ai


@pytest.fixture
def environment(monkeypatch, semantic_stub):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    main.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    calls = []
    for name in ("_require_resource", "_consume_resource", "_persist_offline_knowledge_entry"):
        monkeypatch.setattr(main, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(ai, "_analysis_cache", {})

    async def provider(text="", images_b64=None, category="agriculture"):
        calls.append(text)
        # Deliberately mentions a disease. It must not revive a rejected card.
        return ai._validate_analysis({"type_probleme": category,
            "diagnostic": "Vérifier la situation et ses causes possibles",
            "actions_immediates": ["Observer les chenilles dans le maïs"], "confiance": .6})
    monkeypatch.setattr(ai, "_analyze_unchecked", provider)
    yield db, calls, semantic_stub
    db.close()
    engine.dispose()


def card(db, title, question, answer="Consigne adaptée à la question.", **kwargs):
    item = main.ExpertLocalKnowledgeDB(title=title, question_fr=question,
        resolution_fr=answer, category="agriculture", status="validated", tags_json="[]", **kwargs)
    db.add(item)
    db.commit()
    return item


def query(db, text):
    return asyncio.run(main.v2_assistant_query(main.V2AnalyzeRequest(text=text),
        current_user=SimpleNamespace(id=1, organization_id=None), db=db))


@pytest.mark.parametrize("question, expected", [
    ("Comment cultiver le maïs ?", "Culture du maïs"),
    ("Des termites attaquent mon maïs.", "Termites du maïs"),
    ("Comment conserver le maïs après la récolte ?", "Stockage du maïs"),
    ("Je veux vendre mon maïs.", "Commercialisation du maïs"),
])
def test_same_crop_different_intentions(environment, question, expected):
    db, calls, semantic_calls = environment
    for title, text in [
        ("Culture du maïs", "Comment cultiver le maïs ?"),
        ("Termites du maïs", "Les termites attaquent mon maïs"),
        ("Stockage du maïs", "Comment conserver le maïs après récolte ?"),
        ("Commercialisation du maïs", "Comment vendre mon maïs ?"),
        ("Chenille du maïs", "Les chenilles attaquent mon maïs"),
    ]:
        card(db, title, text)
    result = query(db, question)
    assert result["knowledge_card"]["title"] == expected
    assert not calls
    assert [call["phase"] for call in semantic_calls] == ["understand", "validate"]
    assert semantic_calls[1]["queryProfile"]["originalQuery"] == question


def test_termites_rejects_lexically_close_armyworm_and_does_not_rematch_ai(environment):
    db, calls, semantic_calls = environment
    card(db, "Chenilles des jeunes plants de maïs", "Les chenilles attaquent les jeunes pieds de maïs")
    result = query(db, "Des termites attaquent les jeunes pieds de maïs.")
    assert not result.get("knowledge_card")
    assert result["knowledge_mode"] == "external_ai" and len(calls) == 1
    assert len(semantic_calls) == 2


def test_yellowing_does_not_choose_named_disease(environment):
    db, calls, _ = environment
    card(db, "Rouille du maïs", "Les feuilles du maïs ont la rouille")
    card(db, "Maladie fongique du maïs", "Mon maïs a une maladie fongique confirmée")
    result = query(db, "Les feuilles de mon maïs deviennent jaunes.")
    assert not result.get("knowledge_card") and calls


def test_garlic_training_rejects_disease_and_preserves_courses(environment):
    db, calls, _ = environment
    card(db, "Maladies fongiques de l'ail", "Comment traiter la maladie de l'ail ?")
    db.add(main.AcademyCourseDB(title="Cultiver l'ail", crop="ail",
        summary="Apprendre à cultiver l'ail de la plantation à la récolte.", status="published"))
    db.commit()
    result = query(db, "Comment cultiver l'ail ?")
    assert not result.get("knowledge_card")
    assert result["knowledge_mode"] == "course_knowledge" and result["recommended_course"]
    assert not calls


def profile():
    return {"originalQuery": "Comment conserver le maïs après la récolte ?",
            "normalizedQuery": "comment conserver le mais apres la recolte",
            "detectedDomain": "agriculture", "semanticUnderstandingAvailable": True,
            "detectedIntent": "storage", "userGoal": "Conserver la récolte"}


def candidate():
    return {"id": "studio:1", "title": "Conserver le maïs", "question": "Conserver la récolte",
            "answer": "Séchez les grains avant le stockage.", "lexical_score": 99}


def judgment(identifier="studio:1", score=4, **overrides):
    result = {"id": identifier, "score": score, "subjectMatch": True,
        "intentMatch": True, "problemMatch": True, "contextMatch": True,
        "goalMatch": True, "answersQuestion": True,
        "evidence": "Séchez les grains", "reason": "Répond à la conservation"}
    return {**result, **overrides}


@pytest.mark.parametrize("overrides", [
    {"intentMatch": False}, {"problemMatch": False}, {"subjectMatch": False},
    {"contextMatch": False}, {"goalMatch": False}, {"answersQuestion": False},
    {"evidence": "Texte absent de la fiche"}, {"score": 2}, {"score": True},
    {"intentMatch": "true"},
])
def test_high_lexical_score_never_bypasses_semantic_checks(monkeypatch, overrides):
    monkeypatch.setattr(relevance, "complete", lambda *args: {
        "evaluations": [judgment(**overrides)], "ambiguous": False})
    accepted, audit = relevance.validate(profile(), [candidate()])
    assert accepted == [] and audit["fallbackTriggered"]


@pytest.mark.parametrize("evaluations", [[], [judgment("unknown")], [judgment(), judgment()]])
def test_missing_duplicate_or_unknown_ids_fail_closed(monkeypatch, evaluations):
    monkeypatch.setattr(relevance, "complete", lambda *args: {
        "evaluations": evaluations, "ambiguous": False})
    accepted, audit = relevance.validate(profile(), [candidate()])
    assert not accepted and "unavailable" in audit["rejectionReason"]


@pytest.mark.parametrize("checks, expected", [
    ([True] * 6, True), ([True, False, True, True, True, True], False),
    ([True] * 5, False), (["true"] * 6, False),
])
def test_compact_provider_contract_preserves_all_semantic_checks(monkeypatch, checks, expected):
    compact = {"id": "studio:1", "score": 4, "checks": checks,
               "evidence": "Séchez les grains", "reason": "Réponse directe"}
    monkeypatch.setattr(relevance, "complete", lambda *args: {
        "evaluations": [compact], "ambiguous": False})
    accepted, _ = relevance.validate(profile(), [candidate()])
    assert bool(accepted) is expected


@pytest.mark.parametrize("minimum, expected", [(3, True), (4, False)])
def test_configurable_direct_answer_bands(monkeypatch, minimum, expected):
    monkeypatch.setenv("SONGRA_SEMANTIC_MIN_SCORE", str(minimum))
    monkeypatch.setattr(relevance, "complete", lambda *args: {
        "evaluations": [judgment(score=3)], "ambiguous": False})
    accepted, _ = relevance.validate(profile(), [candidate()])
    assert bool(accepted) is expected


def test_invalid_threshold_cannot_authorize_partial_match(monkeypatch):
    monkeypatch.setenv("SONGRA_SEMANTIC_MIN_SCORE", "1")
    with pytest.raises(ValueError):
        relevance.validate(profile(), [candidate()])


def test_validator_outage_uses_ai_without_approximate_card(environment, monkeypatch):
    db, calls, _ = environment
    card(db, "Culture du maïs", "Comment cultiver le maïs ?")
    def unavailable(*args):
        raise TimeoutError("Provider unavailable")
    monkeypatch.setattr(relevance, "complete", unavailable)
    result = query(db, "Comment cultiver le maïs ?")
    assert not result.get("knowledge_card") and result["knowledge_mode"] == "external_ai" and calls


def test_ambiguous_candidates_ask_question_without_billing_or_ai(environment, monkeypatch):
    db, calls, _ = environment
    card(db, "Jaunissement du maïs par carence", "Les feuilles deviennent jaunes")
    card(db, "Jaunissement du maïs après pluie", "Les feuilles deviennent jaunes après la pluie")
    original_complete = relevance.complete
    def ambiguous(system, payload):
        if payload["phase"] == "understand":
            return original_complete(system, payload)
        return {"evaluations": [judgment(c["id"], evidence=c["answer"]) for c in payload["candidates"]],
            "ambiguous": True, "clarifyingQuestion": "Le sol reste-t-il gorgé d'eau ?"}
    monkeypatch.setattr(relevance, "complete", ambiguous)
    billed = []
    monkeypatch.setattr(main, "_consume_resource", lambda *args, **kwargs: billed.append(True))
    result = query(db, "Les feuilles de mon maïs deviennent jaunes.")
    assert result["needs_clarification"] and result["knowledge_mode"] == "needs_clarification"
    assert result["message"] == "Le sol reste-t-il gorgé d'eau ?"
    assert not result.get("knowledge_card") and not result["actions"] and not calls and not billed


def test_legacy_retrieval_has_no_substring_bypass(environment):
    db, _, _ = environment
    db.add(main.KnowledgeItem(domain="agriculture", title="Chenilles du maïs",
        question="Les chenilles attaquent le maïs", answer="Cherchez les chenilles", tags='["maïs"]'))
    db.commit()
    assert main.retrieve_knowledge(db, "agriculture", "Des termites attaquent le maïs") == []


def test_yingr_v3_rejects_wrong_problem_and_cross_domain(environment, monkeypatch):
    import yingr_ai_services as yingr
    monkeypatch.setattr(yingr, "KNOWLEDGE_ITEMS", [{"id": 1, "domain": "agriculture",
        "title": "Chenilles du maïs", "question": "Les chenilles attaquent le maïs",
        "answer": "Cherchez les chenilles", "tags": ["maïs"]}])
    assert yingr.retrieve_rag_context("Des termites attaquent le maïs", "agriculture") == []
    assert yingr.retrieve_rag_context("Des chenilles attaquent le maïs", "elevage") == []
    assert yingr.retrieve_rag_context("", "agriculture") == []


def test_health_domain_alias_preserves_validated_resources(environment):
    db, _, _ = environment
    db.add(main.KnowledgeItem(domain="health", title="Premiers secours pour une coupure",
        question="Comment aider après une coupure ?", answer="Mettre la personne en sécurité.", tags="[]"))
    db.commit()
    selected = main._select_semantic_resources(db, "health", "Comment aider après une coupure ?")
    assert selected["rag"][0]["domain"] == "health"


def test_course_hidden_lessons_cannot_override_irrelevant_summary(environment, monkeypatch):
    db, _, _ = environment
    db.add(main.AcademyCourseDB(title="Conserver le maïs", crop="maïs", summary="Culture et semis du maïs",
        steps_json=json.dumps([{"content": "Conserver le maïs après récolte"}]), status="published"))
    db.commit()
    original_complete = relevance.complete
    def reject(system, payload):
        if payload["phase"] == "understand":
            return original_complete(system, payload)
        assert payload["candidates"][0]["answer"] == "Culture et semis du maïs"
        return {"evaluations": [judgment(payload["candidates"][0]["id"], score=1,
                                         answersQuestion=False)], "ambiguous": False}
    monkeypatch.setattr(relevance, "complete", reject)
    assert main._find_academy_course_match(db, "Comment conserver le maïs après récolte ?",
        domain="agriculture", organization_id=None) is None
