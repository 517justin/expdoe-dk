from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
PROHIBITED = ("MI" + "-6").casefold()
REMOVED_DEPENDENCY = ("pyDOE" + "3").casefold()


def _archive_members(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    with tarfile.open(path, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def _combined_text(members: dict[str, bytes]) -> str:
    return "\n".join(
        payload.decode("utf-8", errors="ignore") for payload in members.values()
    ).casefold()


def test_release_dependency_and_legal_inputs_are_clean():
    inputs = (
        REPOSITORY_ROOT / "requirements.txt",
        REPOSITORY_ROOT / "NOTICE",
        PACKAGE_ROOT / "NOTICE",
        PACKAGE_ROOT / "LICENSE",
        PACKAGE_ROOT / "pyproject.toml",
    )
    text = "\n".join(path.read_text() for path in inputs).casefold()

    assert PROHIBITED not in text
    assert REMOVED_DEPENDENCY not in text
    assert "version = \"0.5.0\"" in text
    assert "scipy>=1.12" in text


@pytest.mark.slow
def test_built_wheel_and_sdist_are_clean_and_complete(tmp_path):
    build_script = (
        "from setuptools.build_meta import build_sdist, build_wheel; "
        f"build_wheel({str(tmp_path)!r}); build_sdist({str(tmp_path)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", build_script],
        cwd=PACKAGE_ROOT,
        env=os.environ.copy(),
        check=True,
    )
    artifacts = sorted(tmp_path.iterdir())

    assert {path.suffix for path in artifacts} == {".gz", ".whl"}
    for artifact in artifacts:
        members = _archive_members(artifact)
        names = tuple(members)
        text = _combined_text(members)

        assert PROHIBITED not in text
        assert REMOVED_DEPENDENCY not in text
        assert any(name.endswith("expdoe_dk/__init__.py") for name in names)
        assert any(
            name.endswith("expdoe_dk/knowledge/registry/providers.py")
            for name in names
        )
        assert any(name.endswith("NOTICE") for name in names)
        assert any(name.endswith("LICENSE") for name in names)
        if artifact.suffix == ".whl":
            metadata = next(
                payload.decode("utf-8")
                for name, payload in members.items()
                if name.endswith(".dist-info/METADATA")
            )
        else:
            metadata = next(
                payload.decode("utf-8")
                for name, payload in members.items()
                if name.endswith("PKG-INFO")
            )
        assert "Version: 0.5.0" in metadata
        assert "Requires-Dist:scipy>=1.12" in metadata.replace(" ", "")
