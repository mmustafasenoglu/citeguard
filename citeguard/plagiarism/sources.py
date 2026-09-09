"""Safe source acquisition and construction of the shared similarity index."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from citeguard.corpus.ingest import ingest_file
from citeguard.corpus.models import CorpusLanguage, CorpusMetadata
from citeguard.extractor import read_paragraphs
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.models import Fingerprint

from .models import PlagiarismConfig, PlagiarismSource

SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".docx"})


@dataclass(frozen=True, slots=True)
class BuiltSources:
    index: SimilarityIndex
    sources: list[PlagiarismSource]
    warnings: list[str]


class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > self.maximum:
            raise urllib.error.HTTPError(newurl, code, "too many redirects", headers, fp)
        _allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _allowed_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Unsupported source URL: {url}")
    host = parsed.hostname.lower()
    if host in {"localhost", "0.0.0.0", "::1"} or host.startswith("127."):
        raise ValueError("Local and loopback source URLs are not allowed")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or default_port)}
    except socket.gaierror as exc:
        raise ValueError(f"Source URL host could not be resolved: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, local, and reserved source addresses are not allowed")


def _strip_html(payload: str) -> str:
    payload = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", payload)
    payload = re.sub(r"(?s)<[^>]+>", " ", payload)
    return re.sub(r"\s+", " ", html.unescape(payload)).strip()


def fetch_url_text(url: str, config: PlagiarismConfig) -> str:
    """Retrieve explicit textual content with strict transport safeguards."""
    if config.offline:
        raise ValueError("--offline forbids --source-url network access")
    _allowed_url(url)
    cache_dir = Path(".local/citeguard/plagiarism-url-cache")
    cache_path = cache_dir / f"{_hash(url)}.json"
    if not config.no_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return str(cached["text"])
    request = urllib.request.Request(url, headers={"User-Agent": "citeguard/1.1"})
    opener = urllib.request.build_opener(_LimitedRedirectHandler(config.url_max_redirects))
    with opener.open(request, timeout=config.url_timeout) as response:
        content_type = response.headers.get_content_type().lower()
        if content_type not in {"text/plain", "text/html", "application/xhtml+xml"}:
            raise ValueError(f"Unsupported source content type: {content_type}")
        payload = response.read(config.url_max_bytes + 1)
        if len(payload) > config.url_max_bytes:
            raise ValueError("Source URL exceeds the configured maximum size")
        charset = response.headers.get_content_charset() or "utf-8"
        decoded = payload.decode(charset, errors="replace")
        text = _strip_html(decoded) if "html" in content_type else decoded
    if not text.strip():
        raise ValueError("Source URL returned no usable text")
    if not config.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"url": url, "text": text}), encoding="utf-8")
    return text


def _metadata(title: str, *, path: str | None = None, url: str | None = None) -> CorpusMetadata:
    return CorpusMetadata(
        title=title,
        source=path or url,
        url=url,
        language=CorpusLanguage.OTHER,
        license="user_supplied_for_comparison",
        similarity_index_allowed=True,
    )


def _paths(corpora: list[Path], explicit: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in [*corpora, *explicit]:
        if root.is_dir():
            files.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
        else:
            files.append(root)
    unique: dict[str, Path] = {}
    for path in files:
        resolved = path.resolve()
        unique[str(resolved)] = resolved
    return list(unique.values())


def build_sources(
    corpora: list[Path],
    explicit: list[Path],
    urls: list[str],
    config: PlagiarismConfig,
    *,
    discovery_queries: list[str] | None = None,
) -> BuiltSources:
    """Build one index while retaining distinct source identities."""
    entries = []
    sources: list[PlagiarismSource] = []
    warnings: list[str] = []
    for path in _paths(corpora, explicit)[: config.max_sources]:
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported source type: {path.suffix or '<none>'}")
        if path.stat().st_size > config.local_source_max_bytes:
            raise ValueError(f"Source exceeds the configured maximum size: {path}")
        raw = "\n\n".join(read_paragraphs(path))
        source_id = f"file:{path}"
        document = ingest_file(
            path,
            doc_id=source_id,
            metadata=_metadata(path.stem, path=str(path)),
        )
        for entry in document.entries:
            fingerprint = Fingerprint(
                points=winnow(generate_shingles(entry.normalized_text)), doc_id=source_id
            )
            entries.append(
                (
                    source_id,
                    entry.normalized_text,
                    fingerprint,
                    entry.metadata,
                    entry.entry_index,
                    entry,
                )
            )
        normalized_hash = _hash(" ".join(e.normalized_text for e in document.entries))
        sources.append(
            PlagiarismSource(
                source_id,
                path.stem,
                "local",
                str(path),
                None,
                (),
                None,
                _hash(raw),
                normalized_hash,
            )
        )

    remaining = max(config.max_sources - len(sources), 0)
    for url in list(dict.fromkeys(urls))[:remaining]:
        try:
            text = fetch_url_text(url, config)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            warnings.append(f"URL source unavailable ({url}): {exc}")
            continue
        source_id = f"url:{_hash(url)[:16]}"
        title = urlparse(url).hostname or url
        from citeguard.corpus.models import CorpusEntry
        from citeguard.corpus.normalize import normalize_corpus_text_with_map
        from citeguard.sentence_splitter import split_sentences

        normalized_parts: list[str] = []
        for idx, sentence in enumerate(split_sentences(text)):
            normalized, offset_map = normalize_corpus_text_with_map(sentence.text)
            if len(normalized) < 10:
                continue
            metadata = _metadata(title, url=url)
            entry = CorpusEntry(
                sentence.text,
                normalized,
                source_id,
                idx,
                metadata,
                sentence.start_offset,
                sentence.end_offset,
                offset_map=offset_map,
            )
            fp = Fingerprint(points=winnow(generate_shingles(normalized)), doc_id=source_id)
            entries.append((source_id, normalized, fp, metadata, idx, entry))
            normalized_parts.append(normalized)
        sources.append(
            PlagiarismSource(
                source_id,
                title,
                "url",
                None,
                url,
                (),
                None,
                _hash(text),
                _hash(" ".join(normalized_parts)),
            )
        )

    remaining = max(config.max_sources - len(sources), 0)
    if config.discover_academic and remaining:
        if config.offline:
            warnings.append("Academic discovery skipped because offline mode is active")
        elif not discovery_queries:
            warnings.append("Academic discovery skipped because no bibliography query was found")
        else:
            from citeguard.cache import FileCache
            from citeguard.retrieval import RetrievalEngine

            retrieval = RetrievalEngine(
                cache=FileCache(".citeguard_cache"),
                use_cache=not config.no_cache,
                offline=False,
            )
            seen: set[str] = set()
            for query in discovery_queries:
                if len(sources) >= config.max_sources:
                    break
                candidates = retrieval.search(query, max_results=min(5, remaining))
                if retrieval.last_result:
                    warnings.extend(retrieval.last_result.provider_errors)
                for candidate in candidates:
                    identity = candidate.doi or candidate.url or candidate.title.casefold()
                    if identity in seen:
                        continue
                    seen.add(identity)
                    source_id = f"academic:{_hash(identity)[:16]}"
                    abstract = (candidate.abstract or "").strip()
                    sources.append(
                        PlagiarismSource(
                            source_id,
                            candidate.title,
                            "academic_abstract" if abstract else "academic_metadata",
                            None,
                            candidate.url,
                            tuple(candidate.authors),
                            candidate.year,
                            _hash(abstract),
                            _hash(abstract.casefold()),
                            textual_content_compared=bool(abstract),
                            source_discovered=True,
                        )
                    )
                    if not abstract:
                        continue
                    from citeguard.corpus.models import CorpusEntry
                    from citeguard.corpus.normalize import normalize_corpus_text_with_map
                    from citeguard.sentence_splitter import split_sentences

                    metadata = CorpusMetadata(
                        title=candidate.title,
                        authors=candidate.authors,
                        year=candidate.year,
                        doi=candidate.doi,
                        source=candidate.source_api,
                        url=candidate.url,
                        language=CorpusLanguage.OTHER,
                        license="provider_returned_abstract",
                        similarity_index_allowed=True,
                    )
                    for idx, sentence in enumerate(split_sentences(abstract)):
                        normalized, offset_map = normalize_corpus_text_with_map(sentence.text)
                        if len(normalized) < 10:
                            continue
                        entry = CorpusEntry(
                            sentence.text,
                            normalized,
                            source_id,
                            idx,
                            metadata,
                            sentence.start_offset,
                            sentence.end_offset,
                            offset_map=offset_map,
                        )
                        fp = Fingerprint(
                            points=winnow(generate_shingles(normalized)), doc_id=source_id
                        )
                        entries.append((source_id, normalized, fp, metadata, idx, entry))
                    if len(sources) >= config.max_sources:
                        break
                remaining = max(config.max_sources - len(sources), 0)
    from .indexing import load_cached_index, save_cached_index

    cached = load_cached_index(sources, config)
    if cached is not None:
        return BuiltSources(cached, sources, warnings)
    index = SimilarityIndex.build(entries)
    if not config.no_cache and index.size:
        save_cached_index(index, sources, config)
    return BuiltSources(index, sources, warnings)
