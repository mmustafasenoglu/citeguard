from citeguard.cache import FileCache


def test_cache_key_changes_with_prompt_version(tmp_path) -> None:
    cache = FileCache(tmp_path)
    a = cache.make_key(operation="match", provider="anthropic", query="x", prompt_version="1")
    b = cache.make_key(operation="match", provider="anthropic", query="x", prompt_version="2")
    assert a != b


def test_cache_round_trip(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key = cache.make_key(operation="search", provider="crossref", query="transformers")
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}


def test_cache_returns_none_for_missing_key(tmp_path) -> None:
    cache = FileCache(tmp_path)
    result = cache.get("nonexistent_key")
    assert result is None


def test_cache_returns_none_for_corrupt_file(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key = cache.make_key(operation="search", provider="crossref", query="test")
    cache_path = tmp_path / f"{key}.json"
    cache_path.write_text("not valid json {{{", encoding="utf-8")
    assert cache.get(key) is None


def test_cache_overwrites_existing_entry(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key = cache.make_key(operation="search", provider="crossref", query="test")
    cache.set(key, [{"title": "First"}])
    cache.set(key, [{"title": "Second"}])
    assert cache.get(key) == [{"title": "Second"}]


def test_cache_creates_directory(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    cache = FileCache(nested)
    key = cache.make_key(operation="search", provider="crossref", query="test")
    cache.set(key, [1, 2, 3])
    assert cache.get(key) == [1, 2, 3]


def test_cache_key_changes_with_provider(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key1 = cache.make_key(operation="search", provider="crossref", query="test")
    key2 = cache.make_key(operation="search", provider="arxiv", query="test")
    assert key1 != key2


def test_cache_key_changes_with_operation(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key1 = cache.make_key(operation="search", provider="crossref", query="test")
    key2 = cache.make_key(operation="verify", provider="crossref", query="test")
    assert key1 != key2


def test_cache_key_changes_with_query(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key1 = cache.make_key(operation="search", provider="crossref", query="test1")
    key2 = cache.make_key(operation="search", provider="crossref", query="test2")
    assert key1 != key2


def test_cache_key_deterministic(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key1 = cache.make_key(operation="search", provider="crossref", query="test")
    key2 = cache.make_key(operation="search", provider="crossref", query="test")
    assert key1 == key2


def test_cache_handles_json_null(tmp_path) -> None:
    cache = FileCache(tmp_path)
    key = "test_key"
    (tmp_path / f"{key}.json").write_text("null", encoding="utf-8")
    assert cache.get(key) is None
