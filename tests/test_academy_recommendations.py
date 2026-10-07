import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main


@pytest.fixture
def db(semantic_stub):
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    main.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def recommend(db, question="Comment cultiver le maïs ?", org=None):
    return asyncio.run(main.recommend_academy_courses(
        main.AcademyRecommendationRequest(question=question),
        current_user=SimpleNamespace(id=1, organization_id=org), db=db))["courses"]


def add(db, title="Culture du maïs", **kwargs):
    course = main.AcademyCourseDB(title=title, summary="Apprendre la culture du maïs de la plantation à la récolte.",
        crop="maïs", status="published", **kwargs)
    db.add(course)
    db.commit()
    return course


def test_real_matching_course_without_unlock(db):
    course = add(db)
    result = recommend(db)
    assert result[0]["id"] == course.id
    assert result[0]["title"] == course.title
    assert result[0]["summary"] == course.summary
    assert result[0]["open_path"] == f"/academy/courses/{course.id}"
    assert db.query(main.CourseAccessDB).count() == 0


def test_maximum_three_semantic_matches(db):
    for i in range(5):
        add(db, title=f"Culture du maïs {i}")
    courses = recommend(db)
    assert len(courses) == 3
    assert [c['match_score'] for c in courses] == sorted([c['match_score'] for c in courses], reverse=True)


def test_no_course_and_same_word_wrong_intent(db):
    assert recommend(db) == []
    add(db, title="Maladie du maïs")
    assert recommend(db) == []


def test_disease_does_not_search_courses(db, semantic_stub):
    add(db)
    assert recommend(db, "Les feuilles de mon maïs deviennent jaunes") == []
    assert semantic_stub == []


def test_only_available_courses_in_user_scope(db):
    add(db, organization_id=7)
    draft = add(db)
    draft.status = 'draft'
    db.commit()
    assert recommend(db) == []
    assert len(recommend(db, org=7)) == 1


@pytest.mark.parametrize('question, domain', [
    ('Comment cultiver la patate ?', 'agriculture'),
    ('Comment faire du compost ?', 'agriculture'),
    ('Comment élever des poulets ?', 'elevage'),
    ('Comment conserver mes récoltes ?', 'agriculture'),
    ('Comment apprendre à sécuriser mon téléphone ?', 'cybersecurity'),
])
def test_dynamic_learning_intention(question, domain):
    assert main._is_learning_question(question, domain)
