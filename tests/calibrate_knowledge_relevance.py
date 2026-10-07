"""Explicit live calibration, never invoked by pytest; no database writes.

Run from backend: python tests/calibrate_knowledge_relevance.py --output report.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import knowledge_relevance as relevance

CARDS = [
    ("culture", "Cultiver le maïs", "Comment cultiver le maïs ?", "Choisissez la parcelle, préparez le sol, semez après une pluie utile, entretenez la culture et récoltez les épis mûrs."),
    ("termite", "Termites des jeunes plants de maïs", "Des termites attaquent mes jeunes pieds de maïs", "Inspectez les pieds de maïs et les galeries des termites. Faites identifier le ravageur avant de demander une méthode de lutte adaptée aux termites."),
    ("storage", "Conserver le maïs après la récolte", "Comment conserver le maïs contre les insectes du stock ?", "Séchez soigneusement les grains après la récolte. Stockez le maïs sec dans des sacs propres sur des palettes et vérifiez régulièrement les insectes dans le stock."),
    ("armyworm", "Chenille légionnaire du maïs", "Des chenilles mangent les jeunes plants de maïs", "Recherchez les chenilles dans le cœur des feuilles. Retirez les chenilles visibles et demandez conseil si l'attaque s'étend."),
    ("rot", "Pourriture des tiges du maïs", "Mes tiges de maïs pourrissent", "Observez les lésions des tiges et demandez l'identification de la pourriture au conseiller agricole."),
    ("market", "Commercialisation du maïs", "Je veux vendre mon maïs", "Comparez les prix du maïs sur plusieurs marchés, pesez vos sacs et négociez avec des acheteurs fiables avant la vente."),
    ("nitrogen", "Carence en azote du maïs", "Mon maïs a une carence en azote confirmée", "En cas de carence en azote confirmée, demandez au conseiller agricole un apport adapté. Vérifiez les quantités recommandées localement."),
    ("garlic_disease", "Maladies fongiques de l'ail", "L'ail est atteint de maladie fongique", "Faites identifier la maladie fongique de l'ail avant de choisir un traitement adapté."),
    ("garlic_culture", "Culture de l'ail", "Comment cultiver l'ail ?", "Choisissez des caïeux sains, préparez un sol drainant, plantez les caïeux puis arrosez et désherbez pour réussir la culture de l'ail."),
]
CASES = [
    ("A", "Comment cultiver le maïs ?", {"culture"}),
    ("B", "Des termites attaquent mon maïs.", {"termite"}),
    ("C", "Comment conserver le maïs après la récolte ?", {"storage"}),
    ("D", "Les feuilles de mon maïs deviennent jaunes.", set()),
    ("E", "Je veux vendre mon maïs.", {"market"}),
    ("F", "Des termites attaquent les jeunes plants de maïs, pas des chenilles.", {"termite"}),
    ("G", "Comment assurer financièrement mon maïs contre la grêle ?", set()),
    ("H", "Comment cultiver l'ail ?", {"garlic_culture"}),
    ("I", "Comment conserver le maïs après la récolte pour éviter les insectes ?", {"storage"}),
    ("J", "Mes pieds de maïs jaunissent après de fortes pluies, qu'est-ce que je dois faire ?", set()),
]


def threshold_stats(samples):
    comparisons = []
    for minimum in (3, 4):
        false_positive = false_negative = 0
        for sample in samples:
            actual = {e["id"] for e in sample["audit"]["semanticValidationResult"]
                      if e.get("accepted") and e.get("score", 0) >= minimum}
            expected = set(sample["expected"])
            false_positive += len(actual - expected)
            false_negative += len(expected - actual)
        comparisons.append({"threshold": minimum, "falsePositives": false_positive,
                            "falseNegatives": false_negative})
    return comparisons


def run(output, case=None, real_only=False):
    # Calibration is deliberately paced; production fails closed on outages.
    original_complete = relevance.complete
    def paced_complete(system, payload):
        for attempt in range(5):
            try:
                return original_complete(system, payload)
            except Exception as error:
                if type(error).__name__ != "RateLimitError" or attempt == 4:
                    raise
                headers = getattr(getattr(error, "response", None), "headers", {})
                try:
                    delay = min(60, max(5, float(headers.get("retry-after", "20")) + 1))
                except ValueError:
                    delay = 20
                print(json.dumps({"rateLimited": True, "phase": payload["phase"],
                                  "retryInSeconds": delay}), flush=True)
                time.sleep(delay)
    relevance.complete = paced_complete
    samples = []
    for label, query, expected in CASES:
        if real_only:
            continue
        if case is not None and label != case:
            continue
        profile = relevance.understand(query, "agriculture")
        candidates = [{"id": identifier, "title": title, "question": question,
                       "answer": answer, "lexical_score": 0}
                      for identifier, title, question, answer in CARDS]
        tokens = main._rural_tokens(query + " " + " ".join(profile.get("searchTerms", [])))
        for candidate in candidates:
            candidate["lexical_score"] = (4 * len(tokens & main._rural_tokens(candidate["title"]))
                + 2 * len(tokens & main._rural_tokens(candidate["question"]))
                + .25 * len(tokens & main._rural_tokens(candidate["answer"])))
        candidates.sort(key=lambda c: c["lexical_score"], reverse=True)
        candidates = candidates[:5]
        accepted, audit = relevance.validate(profile, candidates)
        actual = {item["id"] for item in accepted}
        available = profile.get("semanticUnderstandingAvailable") and "unavailable" not in audit.get("rejectionReason", "")
        samples.append({"case": label, "expected": sorted(expected), "actual": sorted(actual),
                        "passed": bool(available and actual == expected), "audit": audit})
        print(json.dumps({"case": label, "expected": sorted(expected), "actual": sorted(actual),
                          "passed": bool(available and actual == expected)}, ensure_ascii=True), flush=True)
    if case is not None:
        report = json.loads(Path(output).read_text(encoding="utf-8"))
        report.setdefault("rechecks", []).append({"timestamp": datetime.now(timezone.utc).isoformat(),
            "profilePromptHash": hashlib.sha256(relevance.PROFILE_PROMPT.encode()).hexdigest(),
            "validationPromptHash": hashlib.sha256(relevance.VALIDATION_PROMPT.encode()).hexdigest(),
            "sample": samples[0]})
        latest = {s["case"]: s for s in report["syntheticCases"]}
        for recheck in report["rechecks"]:
            latest[recheck["sample"]["case"]] = recheck["sample"]
        report["postRecheckSummary"] = {"passedCases": sum(s["passed"] for s in latest.values()),
            "totalCases": len(latest), "thresholdComparison": threshold_stats(latest.values()),
            "note": "Initial unchanged cases plus targeted rechecks; not a new complete run."}
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["postRecheckSummary"]), flush=True)
        return
    previous_report = json.loads(Path(output).read_text(encoding="utf-8")) if real_only else {}
    if real_only:
        samples = previous_report["syntheticCases"]
    # Measure the former raw lexical score on the REAL local corpus (read-only).
    database = Path(main.__file__).parent / "resolvehub.db"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    rows = connection.execute("SELECT id,title,question,answer,tags FROM knowledge_items WHERE domain='agriculture'").fetchall()
    distributions = []
    for label, query, expected in CASES[:7]:
        query_tokens = main._rural_tokens(query)
        ranked = []
        for identifier, title, question, answer, tags in rows:
            tags = json.loads(tags or "[]")
            score = (3 * len(query_tokens & set(main._tokenize(title or "")))
                + 2.5 * len(query_tokens & set(main._tokenize(" ".join(tags))))
                + 2 * len(query_tokens & set(main._tokenize(question or "")))
                + .3 * len(query_tokens & set(main._tokenize(answer or ""))))
            ranked.append({"id": identifier, "title": title, "lexical_score": round(score, 2)})
        ranked.sort(key=lambda item: item["lexical_score"], reverse=True)
        by_id = {row[0]: row for row in rows}
        profile = relevance.understand(query, "agriculture")
        tokens = main._rural_tokens(query + " " + " ".join(profile.get("searchTerms", [])))
        current_ranked = []
        for identifier, title, question, answer, tags in rows:
            score = (4 * len(tokens & main._rural_tokens(title or ""))
                + 3.5 * len(tokens & main._rural_tokens(" ".join(json.loads(tags or "[]"))))
                + 2 * len(tokens & main._rural_tokens(question or ""))
                + .25 * len(tokens & main._rural_tokens(answer or "")))
            current_ranked.append({"id": identifier, "title": title, "lexical_score": score})
        current_ranked.sort(key=lambda item: item["lexical_score"], reverse=True)
        shortlist = current_ranked[:5]
        candidates = [{"id": str(item["id"]), "title": item["title"],
                       "question": by_id[item["id"]][2] or "", "answer": by_id[item["id"]][3],
                       "lexical_score": item["lexical_score"]} for item in shortlist]
        accepted, audit = relevance.validate(profile, candidates)
        distributions.append({"case": label, "rawScoreDistribution": ranked,
                              "currentScoreDistribution": current_ranked, "audit": audit,
                              "acceptedIds": [item["id"] for item in accepted]})
        print(json.dumps({"realCorpusCase": label, "accepted": [item["id"] for item in accepted]}, ensure_ascii=True), flush=True)
    connection.close()
    scores = [e["score"] for sample in samples for e in sample["audit"]["semanticValidationResult"]]
    threshold_comparison = threshold_stats(samples)
    report = {**previous_report, "provider": main.v2_services.AI_PROVIDER, "model": main.v2_services.GROQ_MODEL,
        "rubric": "ordinal 0..4, direct-answer bands only", "syntheticCases": samples,
        "realCorpus": {"database": database.name, "agricultureCards": len(rows), "cases": distributions},
        "scoreHistogram": {str(score): scores.count(score) for score in range(5)},
        "thresholdComparison": threshold_comparison,
        "limitations": "Small local calibration, not proof of production accuracy or probability calibration; production VM corpus not available."}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": sum(s["passed"] for s in samples), "total": len(samples),
                      "thresholdComparison": threshold_comparison}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=[case[0] for case in CASES])
    parser.add_argument("--real-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, arguments.case, arguments.real_only)
