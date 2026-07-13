import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def test_artifact_manifest_is_explicit_and_machine_readable():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["release_ready"] is True
    assert data["artifacts"]

    for artifact in data["artifacts"]:
        assert artifact["size_bytes"] > 0
        assert artifact["access"] == "private_on_request"
        assert artifact["reproduction_command"]
        assert "drive_file_id" not in artifact
        assert "url" not in artifact
        if artifact["sha256"] is not None:
            assert SHA256.fullmatch(artifact["sha256"])


def test_private_access_policy_is_actionable_and_does_not_leak_links():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    policy = data["access_policy"]
    assert policy["mode"] == "available_upon_reasonable_academic_request"
    assert EMAIL.fullmatch(policy["contact_email"])
    assert set(policy["request_information"]) == {
        "name",
        "affiliation",
        "intended_reproducibility_use",
    }

    public_text = MANIFEST.read_text(encoding="utf-8")
    assert "drive.google.com" not in public_text
    assert "drive_file_id" not in public_text
