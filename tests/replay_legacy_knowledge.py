"""Read-only reproduction of former selection against the calibration corpus."""
import ast
import json
from pathlib import Path
import subprocess
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from calibrate_knowledge_relevance import CASES


def run():
    backend = Path(main.__file__).parent
    baseline = "fc85d24"
    source = subprocess.run(["git", "-C", str(backend), "show", f"{baseline}:main.py"],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    namespace = dict(vars(main))
    functions = {"_resource_relevant", "_find_studio_knowledge_match", "_find_academy_course_match",
                 "retrieve_knowledge", "resolve_knowledge_answer"}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "legacy-baseline", "exec"), namespace)
    database = backend / "resolvehub.db"
    engine = create_engine(f"sqlite:///file:{database.as_posix()}?mode=ro&uri=true")
    db = sessionmaker(bind=engine)()
    results = []
    for label, question, _ in CASES[:7]:
        answer = namespace["resolve_knowledge_answer"](db, "agriculture", question, allow_external=False)
        item = (answer.get("rag_items") or [None])[0]
        results.append({"case": label, "question": question, "mode": answer["knowledge_mode"],
                        "title": item.get("title") if item else None, "id": item.get("id") if item else None})
    count = db.query(main.KnowledgeItem).count()
    db.close()
    engine.dispose()
    report = {"commit": baseline, "database": database.name, "knowledgeCards": count, "results": results}
    (backend / "tests/relevance_legacy_baseline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
