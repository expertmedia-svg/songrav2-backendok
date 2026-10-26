import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
import v2_services as ai
import yingr_ai_services as yingr


def test_groq_mode_blocks_gemini_clients_and_media(monkeypatch):
    monkeypatch.setattr(ai, 'AI_PROVIDER', 'groq')
    for factory in (ai._get_model, ai._get_media_client):
        with pytest.raises(RuntimeError, match='Groq exclusif'):
            factory()
    assert asyncio.run(ai.generate_image('Une feuille'))['success'] is False
    video = asyncio.run(ai.generate_video('Une culture', gemini_api_key=None))
    assert video['success'] is False


def test_entrepreneurship_uses_groq_without_provider_fallback(monkeypatch):
    monkeypatch.setattr(ai, 'AI_PROVIDER', 'groq')
    calls = []
    async def chat(system, prompt, images, **kwargs):
        calls.append((prompt, images))
        return json.dumps({'description_terrain': 'Terrain agricole'})
    monkeypatch.setattr(ai, '_groq_chat', chat)
    monkeypatch.setattr(ai, '_validate_entrepreneurship', lambda value: value)
    result = asyncio.run(ai.gemini_analyze_entrepreneurship('Terrain de deux hectares', ['photo']))
    assert result['description_terrain'] == 'Terrain agricole'
    assert calls[0][1] == ['photo']


@pytest.mark.parametrize('text', ['Les termites attaquent le mais', ''])
def test_groq_whisper_preserves_upload_and_rejects_empty_transcript(monkeypatch, text):
    monkeypatch.setenv('AI_PROVIDER', 'groq')
    received = []
    def transcribe(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(text=text)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe)))
    monkeypatch.setattr(ai, '_get_groq_client', lambda: client)
    if text:
        assert asyncio.run(yingr.transcribe_audio_whisper(b'uploaded-audio', 'voice.wav')) == text
    else:
        with pytest.raises(RuntimeError, match='indisponible'):
            asyncio.run(yingr.transcribe_audio_whisper(b'uploaded-audio', 'voice.wav'))
    assert received[0]['file'] == ('voice.wav', b'uploaded-audio')
    assert received[0]['model'] == 'whisper-large-v3-turbo'


def test_v3_inference_uses_groq_with_image(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'groq')
    calls = []
    async def chat(system, prompt, images, **kwargs):
        calls.append((prompt, images, kwargs))
        return '{"status":"ok"}'
    monkeypatch.setattr(ai, '_groq_chat', chat)
    assert asyncio.run(yingr.run_yingr_ai_inference('Diagnostiquer', 'agriculture', 'photo')) == '{"status":"ok"}'
    assert calls == [('Diagnostiquer', ['photo'], {'json_mode': True})]


def test_dashboard_exists_and_requires_authentication():
    with TestClient(main.app) as client:
        assert client.get('/api/dashboard').status_code == 401


def test_health_identifies_groq(monkeypatch):
    monkeypatch.setattr(ai, 'AI_PROVIDER', 'groq')
    with TestClient(main.app) as client:
        data = client.get('/api/v2/health').json()
    assert data['provider'] == 'groq'
    assert data['model'] == ai.GROQ_MODEL
