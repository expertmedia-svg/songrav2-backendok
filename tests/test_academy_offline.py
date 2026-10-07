import asyncio
import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import academy_offline
import main


def course():
    return {'id': 42, 'title': '[TEST] Cours visuel', 'summary': 'Test uniquement',
        'course_type': 'culture', 'crop': 'Maïs', 'status': 'published',
        'updated_at': '2026-10-07', 'cover_url': 'https://vm.test/uploads/step.webp',
        'steps': [{'id': 'terrain', 'title': 'Terrain', 'content': 'Contenu de test',
            'image_url': 'https://vm.test/uploads/step.webp',
            'audio': {'moore': {'url': 'https://vm.test/uploads/moore.mp3'}}}], 'audio': {}}


def test_real_sizes_hashes_and_unique_assets(tmp_path):
    (tmp_path / 'step.webp').write_bytes(b'real image bytes')
    (tmp_path / 'moore.mp3').write_bytes(b'real recorded audio bytes')
    manifest = academy_offline.build_manifest(course(), tmp_path, 'https://vm.test')
    assert len(manifest['assets']) == 2
    assert manifest['media_size_bytes'] == sum(p.stat().st_size for p in tmp_path.iterdir())
    assert manifest['assets'][0]['sha256'] == hashlib.sha256(b'real image bytes').hexdigest()
    assert manifest['languages'] == ['moore']
    assert manifest['visual_complete']
    assert manifest['course']['version'] == academy_offline.version_for(course())


def test_version_changes_when_content_media_or_audio_changes():
    first = course()
    second = course()
    second['steps'][0]['content'] = 'Nouveau contenu validé'
    assert academy_offline.version_for(first) != academy_offline.version_for(second)
    second = course()
    second['steps'][0]['audio']['fr'] = {'url': 'https://vm.test/uploads/new.mp3'}
    assert academy_offline.version_for(first) != academy_offline.version_for(second)


def test_serialized_version_is_stable_with_legacy_undated_audio():
    from datetime import datetime
    item = SimpleNamespace(id=42, title='[TEST] Cours', course_type='culture', crop='Maïs',
        summary='Test', cover_url=None, steps_json='[{"id":"one","title":"Étape","audio":{}}]',
        audio_json='{"gourmantche":{"url":"https://vm.test/voice.mp3"}}', status='published',
        organization_id=None, created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    one = main._serialize_academy_course(item)
    two = main._serialize_academy_course(item, include_content=False)
    assert one['version'] == two['version'] == academy_offline.version_for(one)
    assert 'gourmantche' in one['audio'] and 'fr' not in one['audio']
    assert 'uploaded_at' not in one['audio']['gourmantche']


def test_external_media_size_stays_unknown_no_server_fetch(tmp_path):
    data = course()
    data['cover_url'] = None
    data['steps'][0]['image_url'] = 'https://elsewhere.test/image.webp'
    data['steps'][0]['audio'] = {}
    result = academy_offline.build_manifest(data, tmp_path, 'https://vm.test')
    assert result['media_size_bytes'] is None
    assert result['assets'][0]['sha256'] is None
    assert result['languages'] == []


@pytest.mark.parametrize('url', ['https://vm.test/uploads/missing.webp',
    'https://vm.test/uploads/%2e%2e/secret', 'file:///etc/passwd'])
def test_missing_or_invalid_file_never_produces_ready_manifest(tmp_path, url):
    data = course()
    data['cover_url'] = url
    with pytest.raises(ValueError):
        academy_offline.build_manifest(data, tmp_path, 'https://vm.test')


def test_empty_steps_are_not_a_fake_visual_course(tmp_path):
    data = course()
    data['steps'] = []
    with pytest.raises(ValueError):
        academy_offline.build_manifest(data, tmp_path, 'https://vm.test')


def test_manifest_preserves_existing_access_and_organization(monkeypatch):
    engine = create_engine('sqlite://')
    main.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    item = main.AcademyCourseDB(title='[TEST] Formation', summary='Test',
        steps_json='[{"id":"one","title":"Étape","content":"Test","audio":{}}]',
        status='published', organization_id=7)
    db.add(item); db.commit()
    user = SimpleNamespace(id=1, organization_id=7, subscription_started_at=None)
    monkeypatch.setattr(main, '_active_offer', lambda user: (None, None))
    preview = asyncio.run(main.academy_offline_info(item.id, current_user=user, db=db))
    assert preview['id'] == item.id
    assert 'course' not in preview and 'steps' not in preview
    assert all('url' not in asset for asset in preview['assets'])
    assert db.query(main.CourseAccessDB).count() == 0
    with pytest.raises(HTTPException) as denied:
        asyncio.run(main.academy_offline_manifest(item.id, current_user=user, db=db))
    assert denied.value.status_code == 403
    assert db.query(main.CourseAccessDB).count() == 0
    db.add(main.CourseAccessDB(user_id=1, course_id=item.id, permanent=True, source='free'))
    db.commit()
    result = asyncio.run(main.academy_offline_manifest(item.id, current_user=user, db=db))
    assert result['course']['id'] == item.id
    assert db.query(main.CourseAccessDB).count() == 1
    user.organization_id = None
    with pytest.raises(HTTPException) as outside:
        asyncio.run(main.academy_offline_manifest(item.id, current_user=user, db=db))
    assert outside.value.status_code == 404
    db.close(); engine.dispose()


def test_teaching_image_upload_compresses_actual_image_without_new_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = create_engine('sqlite://')
    main.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    item = main.AcademyCourseDB(title='[TEST] Cours', summary='Test', status='draft',
        steps_json='[{"id":"one","title":"Étape","content":"Texte validé","audio":{}}]')
    db.add(item); db.commit()
    image = BytesIO()
    Image.new('RGB', (2400, 1800), 'green').save(image, format='PNG')
    upload = UploadFile(filename='illustration.png', file=BytesIO(image.getvalue()))
    result = asyncio.run(main.upload_academy_course_media(item.id, media_kind='image',
        file=upload, step_id='one', language=None,
        current_expert=SimpleNamespace(role='admin'), db=db))
    step = result['course']['steps'][0]
    assert step['content'] == 'Texte validé'
    assert step['audio'] == {}
    assert result['course']['status'] == 'draft'
    assert step['image_url'].endswith('.webp')
    optimized = list(tmp_path.rglob('*.webp'))[0]
    with Image.open(optimized) as picture:
        assert max(picture.size) <= 1440
        assert picture.format == 'WEBP'
    assert optimized.stat().st_size < len(image.getvalue())
    db.close(); engine.dispose()
