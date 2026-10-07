"""Read-only manifests over the existing Academy course and uploaded media."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit


def language_code(value):
    code = ''.join(c for c in unicodedata.normalize('NFD', str(value).strip().lower())
        if unicodedata.category(c) != 'Mn')
    aliases = {'francais': 'fr', 'french': 'fr', 'more': 'moore', 'mooree': 'moore', 'mossi': 'moore',
        'jula': 'dioula', 'dyula': 'dioula', 'fula': 'fulfulde', 'fulani': 'fulfulde',
        'peul': 'fulfulde', 'peulh': 'fulfulde'}
    code = aliases.get(code, code)
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,31}', code):
        raise ValueError('Identifiant de langue invalide')
    return code


def normalize_audio(raw):
    """Keep real recordings and real dates; never relabel unknown languages as French."""
    result = {}
    for language, value in (raw if isinstance(raw, dict) else {}).items():
        if not isinstance(value, dict) or not str(value.get('url') or '').strip():
            continue
        try:
            key = language_code(language)
        except ValueError:
            continue
        result[key] = {**value, 'url': str(value['url']).strip()}
    return result


def version_for(course):
    fields = {key: course.get(key) for key in (
        "id", "title", "course_type", "crop", "summary", "cover_url", "steps",
        "audio", "status", "organization_id", "updated_at")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def build_manifest(course, uploads_root, public_base):
    course = {**course, "version": version_for(course)}
    assets = {}
    def add(url, kind):
        if not isinstance(url, str) or not url.strip():
            return
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Adresse média du cours invalide")
        asset = {"url": url, "kind": kind, "size_bytes": None, "sha256": None}
        root = Path(uploads_root).resolve()
        if parsed.netloc == urlsplit(public_base).netloc and parsed.path.startswith('/uploads/'):
            path = (root / unquote(parsed.path[len('/uploads/'):])).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Chemin média invalide")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Un média du cours est absent du serveur")
            asset['size_bytes'] = path.stat().st_size
            with path.open('rb') as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(128 * 1024), b''):
                    digest.update(chunk)
                asset['sha256'] = digest.hexdigest()
        assets[url] = asset

    def audios(raw):
        for entry in (raw or {}).values():
            if isinstance(entry, dict):
                add(entry.get('url'), 'audio')

    steps = course.get('steps') or []
    if not steps or any(not isinstance(s, dict) or not s.get('id') or not s.get('title') for s in steps):
        raise ValueError("Ce cours ne contient pas encore d'étapes utilisables")
    if len({s['id'] for s in steps}) != len(steps):
        raise ValueError("Identifiants d'étapes dupliqués")
    add(course.get('cover_url'), 'image')
    audios(course.get('audio'))
    for step in steps:
        add(step.get('image_url'), 'image')
        audios(step.get('audio'))
    known = all(a['size_bytes'] is not None for a in assets.values())
    return {"course": course, "assets": list(assets.values()),
        "media_size_bytes": sum(a['size_bytes'] for a in assets.values()) if known else None,
        "visual_complete": all(bool(s.get('image_url')) for s in steps),
        "languages": sorted({language for s in steps for language, a in (s.get('audio') or {}).items()
            if isinstance(a, dict) and a.get('url')})}
