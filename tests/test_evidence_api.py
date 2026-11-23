import hashlib
import json

import pytest

from siesta_remag.evidence import verify_report


def test_public_reader_checks_source_and_artifact_bytes(tmp_path):
    artifact = tmp_path / "observation.dat"
    artifact.write_bytes(b"example")
    path = tmp_path / "report.json"
    report = {
        "format": "REMAG-EXPERIMENT-RESULT/1",
        "source_sha256": "a" * 64,
        "request": {"source_sha256": "a" * 64},
        "artifacts": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    path.write_text(json.dumps(report))
    assert verify_report(path)["verification"]["mode"] == "full local artifacts"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Calculation artifact binding differs"):
        verify_report(path)
    report["source_sha256"] = "b" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Calculation source binding differs"):
        verify_report(path)
