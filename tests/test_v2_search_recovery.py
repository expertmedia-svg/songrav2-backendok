import asyncio
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import v2_services


@pytest.mark.parametrize("provider", ["openai", "groq"])
def test_analysis_retries_after_cached_fallback(monkeypatch, provider):
    monkeypatch.setattr(v2_services, "AI_PROVIDER", provider)
    key = v2_services._get_cache_key("feuilles jaunes", "agriculture", False)
    monkeypatch.setattr(v2_services, "_analysis_cache", {
        key: {"data": {"from_fallback": True}, "ts": time.time()}
    })
    results = iter([{"from_fallback": True}, {"diagnostic": "Analyse precise"}])
    calls = []

    async def analyze(*args):
        calls.append(args)
        return next(results)

    monkeypatch.setattr(v2_services, f"_{provider}_analyze", analyze)
    for _ in range(3):
        result = asyncio.run(v2_services.gemini_analyze("feuilles jaunes"))
    assert result["diagnostic"] == "Analyse precise"
    assert result["from_cache"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("mode, expected", [
    ("no_match", "Diagnostic precis"),
    ("rag_strict", "Reponse de la fiche"),
])
def test_missing_knowledge_preserves_mobile_analysis(monkeypatch, mode, expected):
    async def analyze(**kwargs):
        return {"diagnostic": "Diagnostic precis"}

    monkeypatch.setattr(main.v2_services, "gemini_analyze", analyze)
    monkeypatch.setattr(main.v2_services, "decide", lambda analysis: {})
    monkeypatch.setattr(main.v2_services, "build_response", lambda **kwargs: {
        "message": "Diagnostic precis", "diagnostic": {"description": "Diagnostic precis"}
    })
    monkeypatch.setattr(main, "_find_studio_knowledge_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "resolve_knowledge_answer", lambda **kwargs: {
        "knowledge_mode": mode, "rag_fallback_answer": "Reponse de la fiche"
    })
    for name in ("_require_resource", "_consume_resource", "_persist_offline_knowledge_entry"):
        monkeypatch.setattr(main, name, lambda *args, **kwargs: None)
    result = asyncio.run(main.v2_assistant_query(
        main.V2AnalyzeRequest(text="Mon mais a des feuilles jaunes", category="agriculture"),
        current_user=SimpleNamespace(id=1, organization_id=None), db=None,
    ))
    assert result["message"] == expected
    assert result["diagnostic"]["description"] == "Diagnostic precis"
