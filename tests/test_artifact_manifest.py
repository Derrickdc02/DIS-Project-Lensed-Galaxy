import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_artifact_manifest_is_explicit_and_machine_readable():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert isinstance(data["release_ready"], bool)
    assert data["artifacts"]

    ids = set()
    for artifact in data["artifacts"]:
        assert artifact["drive_file_id"] not in ids
        ids.add(artifact["drive_file_id"])
        assert artifact["size_bytes"] > 0
        assert artifact["visibility"] in {"private", "public"}
        assert artifact["reproduction_command"]
        if artifact["sha256"] is not None:
            assert SHA256.fullmatch(artifact["sha256"])
        if artifact["visibility"] == "public":
            assert artifact["sha256"] is not None
            assert artifact["source_git_sha"] is not None


def test_release_gate_matches_incomplete_artifact_metadata():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    incomplete = any(
        item["visibility"] != "public"
        or item["sha256"] is None
        or item["source_git_sha"] is None
        for item in data["artifacts"]
    )
    assert data["release_ready"] is not incomplete
