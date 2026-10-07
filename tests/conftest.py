"""Controlled semantic-provider contract for offline integration tests.

This test double is not the production algorithm. Live semantic quality is
measured separately by calibrate_knowledge_relevance.py.
"""
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import knowledge_relevance as relevance


@pytest.fixture
def semantic_stub(monkeypatch):
    calls = []

    def facets(text):
        text = relevance.normalized(text)
        if re.search(r"conserv|stock", text):
            intent = "storage"
        elif re.search(r"vend|vente|commercial|marche", text):
            intent = "market"
        elif re.search(r"termite|chenille|armyworm|maladi|fongique|jaun|rouille|tache|meur|attaque", text):
            intent = "diagnostic"
        elif re.search(r"cultiv|culture|sem|plant|calendrier|periode|apprendre", text):
            intent = "cultivation"
        else:
            intent = "advice"
        subject = "mais" if re.search(r"\b(mais|corn|maize)\b", text) else "ail" if re.search(r"\bail\b", text) else ""
        problem = next((p for p in ("termite", "chenille", "rouille", "jaune") if p in text), "")
        if "armyworm" in text:
            problem = "chenille"
        if "taches orange" in text:
            problem = "rouille"
        return intent, subject, problem

    def complete(system, payload):
        calls.append(payload)
        if payload["phase"] == "understand":
            question = payload["originalQuery"]
            observed = payload.get("observations") or {}
            intent, subject, problem = facets(question + " " + " ".join(str(v) for v in observed.values()))
            return {"detectedIntent": intent, "detectedDomain": payload["domain"],
                "detectedCrop": subject or None, "detectedAnimal": None,
                "detectedProblem": problem or None, "detectedSymptoms": [],
                "requestedAction": intent, "userGoal": question or "Analyse de photo",
                "context": "", "searchTerms": [subject, problem, "culture" if intent == "cultivation" else intent]}
        profile = payload["queryProfile"]
        intent, subject, problem = profile["detectedIntent"], profile.get("detectedCrop"), profile.get("detectedProblem")
        evaluations = []
        for c in payload["candidates"]:
            other_intent, other_subject, other_problem = facets(c["title"] + " " + c["question"])
            matches = (intent == other_intent and (not subject or not other_subject or subject == other_subject)
                       and (not problem or problem == other_problem))
            calendar = re.search(r"quand|periode|calendrier", relevance.normalized(profile["originalQuery"]))
            if calendar and not re.search(r"quand|periode|calendrier|date", relevance.normalized(c["title"] + " " + c["answer"])):
                matches = False
            evaluations.append({"id": c["id"], "score": 4 if matches else 1,
                "subjectMatch": matches, "intentMatch": matches, "problemMatch": matches,
                "contextMatch": matches, "goalMatch": matches, "answersQuestion": matches,
                "evidence": c["answer"] if matches else "", "reason": "controlled semantic test judgment"})
        return {"evaluations": evaluations, "ambiguous": False, "clarifyingQuestion": ""}

    monkeypatch.setattr(relevance, "complete", complete)
    return calls
