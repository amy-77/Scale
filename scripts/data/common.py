"""Small shared helpers for reproducible dataset preparation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(record: dict) -> str:
    text = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def download(url: str, path: Path) -> None:
    import requests
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(path)


def hf_endpoint() -> str:
    return os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
