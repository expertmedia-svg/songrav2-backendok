"""Scenarios de mission : vrais moteurs locaux, fournisseurs externes controles."""
import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import v2_services as ai
import yingr_ai_services as yingr


@pytest.fixture
def environment(monkeypatch, semantic_stub):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    main.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    for name in ("_require_resource", "_consume_resource", "_persist_offline_knowledge_entry"):
        monkeypatch.setattr(main, name, lambda *args, **kwargs: None)
    calls = []

    async def provider(text="", images_b64=None, category="agriculture"):
        calls.append((text, images_b64, category))
        return ai._validate_analysis({
            "type_probleme": category, "diagnostic": "Observer le probleme et demander un conseiller si aggravation",
            "confiance": .6, "gravite": "moyenne", "actions_immediates": ["Observer les sujets touches"],
            "image_relevant": True, "image_quality": "good", "description_visuelle": "Sujet visible",
        })

    monkeypatch.setattr(ai, "_analyze_unchecked", provider)
    yield db, calls
    db.close()
    engine.dispose()


def query(db, **kwargs):
    return asyncio.run(main.v2_assistant_query(main.V2AnalyzeRequest(**kwargs),
        current_user=SimpleNamespace(id=1, organization_id=None), db=db))


def fiche(db, **kwargs):
    data = dict(title="Termites du maïs", category="agriculture",
        question_fr="Les termites attaquent les jeunes plants de maïs",
        resolution_fr="Observez les pieds et demandez un conseil agricole local.",
        tags_json=json.dumps(["termite", "maïs", "ravageur"]), status="validated")
    data.update(kwargs)
    item = main.ExpertLocalKnowledgeDB(**data)
    db.add(item)
    db.commit()
    return item


def photo():
    buffer = BytesIO()
    image = Image.new("RGB", (128, 128), (30, 100, 40))
    for x in range(64, 128):
        for y in range(128):
            image.putpixel((x, y), (120, 160, 80))
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_01_termite_text(environment):
    db, calls = environment
    result = query(db, text="Il y a des termites dans mon champ de maïs.")
    assert result["question_intent"] == "plant_disease"
    assert result["message"] and calls


def test_02_voice_transcription_enters_same_engine(environment):
    db, calls = environment
    result = query(db, text="Il y a des termites dans mon champ de maïs.", input_type="voice", source_lang="fr")
    assert result["input_type"] == "voice" and "termites" in calls[0][0]


def test_03_exact_fiche_precedes_external_ai(environment):
    db, calls = environment
    item = fiche(db)
    result = query(db, text="Termites du maïs")
    assert result["knowledge_card"]["id"] == item.id
    assert result["message"] == item.resolution_fr and not calls


@pytest.mark.parametrize("text", ["Les termites mangent mes pieds de maïs", "Mes jeunes plants de maïs sont attaqués par des termites"])
def test_04_equivalent_formulations(environment, text):
    db, calls = environment
    item = fiche(db)
    assert query(db, text=text)["knowledge_card"]["id"] == item.id
    assert not calls


def test_05_garlic_knowledge_and_course(environment):
    db, calls = environment
    item = fiche(db, title="Culture de l'ail", question_fr="Comment cultiver l'ail ?",
        tags_json='["ail", "culture"]', resolution_fr="Choisissez un terrain adapte et des caieux sains.")
    db.add(main.AcademyCourseDB(title="Culture de l'ail", crop="ail", summary="Apprendre la culture de l'ail", status="published"))
    db.commit()
    result = query(db, text="Comment cultiver l’ail ?")
    assert result["knowledge_card"]["id"] == item.id
    assert result["recommended_course"]["id"] and not calls


def test_06_course_only_uses_real_summary(environment):
    db, calls = environment
    course = main.AcademyCourseDB(title="Culture de l'ail", crop="ail", summary="Choisir les caieux sains pour la culture de l'ail.", status="published")
    db.add(course)
    db.commit()
    result = query(db, text="Comment cultiver l'ail ?")
    assert result["recommended_course"]["id"] == course.id
    assert result["message"] == course.summary and not calls


def test_07_no_course_still_answers(environment):
    db, calls = environment
    result = query(db, text="Comment cultiver l'ail ?")
    assert result["recommended_course"] is None and result["message"] and calls


def test_08_poultry_mortality(environment):
    db, calls = environment
    result = query(db, text="Mes poulets meurent.")
    assert result["category"] == "elevage"
    assert result["question_intent"] == "animal_disease"


def test_09_local_language_preserved(environment, monkeypatch):
    import burkina_translator
    db, calls = environment
    monkeypatch.setattr(burkina_translator, "translate_query_to_french", lambda *args: {
        "french_query": "Les termites attaquent mon maïs", "confidence": .9, "source": "verified_stt_alias"})
    result = query(db, text="question locale", source_lang="moore", target_lang="moore", input_type="voice")
    assert result["source_lang"] == "moore" and result["target_lang"] == "moore"
    assert result["original_query"] == "question locale" and "termites" in calls[0][0]


def test_10_local_audio(environment):
    db, calls = environment
    fiche(db, audio_json=json.dumps({"moore": {"url": "/uploads/moore.mp3", "mime_type": "audio/mpeg"}}))
    result = query(db, text="Termites du maïs", target_lang="moore")
    assert result["audio_url"] == "/uploads/moore.mp3" and result["local_audio_available"]


def test_11_french_audio_offer(environment):
    db, calls = environment
    fiche(db, audio_json=json.dumps({"fr": {"url": "/uploads/fr.mp3"}}))
    result = query(db, text="Termites du maïs", target_lang="dioula")
    assert not result["local_audio_available"] and result["audio_url"] is None
    assert result["french_fallback_available"]
    assert result["localizations"]["fr"]["audio_url"] == "/uploads/fr.mp3"


@pytest.mark.parametrize("category", ["agriculture", "elevage"])
def test_12_13_valid_image_analysis(environment, category):
    db, calls = environment
    result = query(db, photo_base64=photo(), category=category)
    assert result["category"] == category and not result["needs_clarification"]
    assert calls[0][1]


@pytest.mark.parametrize("quality, relevant", [("good", False), ("blurry", True)])
def test_14_15_wrong_or_blurry_photo(environment, monkeypatch, quality, relevant):
    db, calls = environment
    fiche(db)
    async def provider(*args):
        return ai._validate_analysis({"diagnostic": "Maladie inventee", "image_relevant": relevant, "image_quality": quality})
    monkeypatch.setattr(ai, "_analyze_unchecked", provider)
    result = query(db, text="Termites du maïs", photo_base64=photo())
    assert result["needs_clarification"] and "Maladie inventee" not in result["message"]
    assert not result.get("knowledge_card") and not result["actions"]


def test_16_photo_and_text_together(environment):
    db, calls = environment
    query(db, text="Pourquoi cette feuille devient jaune ?", photo_base64=photo())
    assert "feuille" in calls[0][0] and calls[0][1]


def test_17_missing_knowledge_external_fallback(environment):
    db, calls = environment
    result = query(db, text="Comment conserver mes récoltes ?")
    assert result["knowledge_mode"] == "external_ai" and len(calls) == 1


def test_18_provider_down_no_false_diagnosis(environment, monkeypatch):
    db, calls = environment
    async def provider(*args):
        return ai._build_analysis_fallback("", "agriculture", False, RuntimeError("indisponible"))
    monkeypatch.setattr(ai, "_analyze_unchecked", provider)
    result = query(db, text="Comment cultiver l'ail ?")
    assert result["knowledge_mode"] == "ai_unavailable"
    assert "indisponible" in result["message"] and not result["actions"]


def test_19_followup_keeps_subject(environment):
    db, calls = environment
    query(db, text="Elles attaquent surtout les jeunes pieds.", conversation_context=[
        {"role": "user", "content": "Mon maïs est attaqué par des termites."},
        {"role": "assistant", "content": "Quels pieds sont touches ?"}])
    assert "termites" in calls[0][0] and "maïs" in calls[0][0]


def test_no_fake_whisper_transcript(monkeypatch):
    monkeypatch.setattr(yingr, "YINGR_AI_WHISPER_URL", "")
    with pytest.raises(RuntimeError, match="indisponible"):
        asyncio.run(yingr.transcribe_audio_whisper(b"audio"))


def test_no_fake_yingr_inference(monkeypatch):
    monkeypatch.setattr(yingr, "YINGR_AI_VLLM_TEXT_URL", "")
    with pytest.raises(RuntimeError, match="indisponible"):
        asyncio.run(yingr.run_yingr_ai_inference("question", "agriculture"))


def test_empty_voice_rejected(environment):
    db, _ = environment
    with pytest.raises(main.HTTPException) as error:
        query(db, text="", input_type="voice")
    assert error.value.status_code == 422


def test_single_crop_does_not_pick_disease(environment):
    db, _ = environment
    fiche(db, title="Rouille du maïs", question_fr="Taches orange de rouille", tags_json='["rouille", "maïs"]')
    result = query(db, text="Comment semer mon maïs ?")
    assert not result.get("knowledge_card")


def test_invalid_image_never_calls_provider(environment):
    db, calls = environment
    result = query(db, photo_base64="not-an-image")
    assert result["needs_clarification"] and not calls


def test_json_validation_handles_zero_confidence_and_null_arrays():
    result = ai._validate_analysis({"diagnostic": "Observation incertaine", "confiance": 0, "causes_probables": None})
    assert result["confiance"] == 0 and result["causes_probables"] == []
    with pytest.raises(ValueError):
        ai._validate_analysis({})


def test_dark_photo_does_not_call_provider(environment):
    db, calls = environment
    output = BytesIO()
    Image.new("RGB", (128, 128), "black").save(output, format="PNG")
    result = query(db, photo_base64=base64.b64encode(output.getvalue()).decode())
    assert result["needs_clarification"] and not calls


def test_missing_french_audio_is_not_claimed(environment):
    db, _ = environment
    fiche(db)
    result = query(db, text="Termites du maïs", target_lang="dioula")
    assert not result["french_fallback_available"]


def test_uncertain_local_transcript_requests_repeat(environment, monkeypatch):
    import burkina_translator
    db, calls = environment
    monkeypatch.setattr(burkina_translator, "translate_query_to_french", lambda *args: {
        "french_query": "mots incertains", "confidence": .2})
    with pytest.raises(main.HTTPException) as error:
        query(db, text="sons incertains", input_type="voice", source_lang="moore")
    assert error.value.status_code == 422 and not calls


def test_knowledge_failure_still_calls_ai(environment, monkeypatch):
    db, calls = environment
    def broken(**kwargs):
        raise RuntimeError("knowledge unavailable")
    monkeypatch.setattr(main, "resolve_knowledge_answer", broken)
    result = query(db, text="Comment cultiver l'ail ?")
    assert result["knowledge_mode"] == "external_ai" and calls


def test_provider_exception_is_clean_and_not_billed(environment, monkeypatch):
    db, calls = environment
    billed = []
    async def broken(*args):
        raise TimeoutError("timeout")
    monkeypatch.setattr(ai, "_analyze_unchecked", broken)
    monkeypatch.setattr(main, "_consume_resource", lambda *args: billed.append(args))
    result = query(db, text="Comment cultiver l'ail ?")
    assert result["knowledge_mode"] == "ai_unavailable" and not billed


def test_learning_answer_is_not_presented_as_disease(environment):
    db, _ = environment
    result = query(db, text="Je veux apprendre la culture de l'ail")
    assert "Diagnostic agricole" not in result["message"]
    assert "Etapes" in result["message"]


@pytest.mark.parametrize("transcript", ["Les termites attaquent mon maïs", ""])
def test_whisper_receives_uploaded_audio_and_rejects_empty_result(monkeypatch, transcript):
    received = []
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            received.append((url, kwargs))
            return SimpleNamespace(status_code=200, json=lambda: {"text": transcript})
    monkeypatch.setattr(yingr.httpx, "AsyncClient", Client)
    monkeypatch.setattr(yingr, "YINGR_AI_WHISPER_URL", "http://whisper.test/v1")
    if transcript:
        assert asyncio.run(yingr.transcribe_audio_whisper(b"real-upload-bytes", "voice.wav")) == transcript
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(yingr.transcribe_audio_whisper(b"real-upload-bytes", "voice.wav"))
    assert received[0][0] == "http://whisper.test/v1/audio/transcriptions"
    assert received[0][1]["files"]["file"][1] == b"real-upload-bytes"


def test_french_text_with_local_audio_preference_is_not_translated(environment, monkeypatch):
    import burkina_translator
    db, calls = environment
    def forbidden(*args):
        raise AssertionError("French question must not be retranscribed")
    monkeypatch.setattr(burkina_translator, "translate_query_to_french", forbidden)
    result = query(db, text="Comment cultiver l'ail ?", source_lang="moore", target_lang="moore")
    assert result["source_lang"] == "fr" and result["target_lang"] == "moore"


def test_local_typed_question_is_normalized(environment, monkeypatch):
    import burkina_translator
    db, calls = environment
    monkeypatch.setattr(burkina_translator, "translate_query_to_french", lambda *args: {
        "french_query": "Comment cultiver l'ail", "confidence": .9})
    result = query(db, text="question locale", source_lang="dioula", target_lang="dioula")
    assert result["source_lang"] == "dioula" and "cultiver" in calls[0][0]
