"""The release workflow's tag check (scripts/check_release_tag.py) and its publishing gate."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_release_tag  # noqa: E402


@pytest.mark.parametrize(("tag", "version"), [
    ("v0.3.0-alpha", "0.3.0a0"), ("v0.3.0-alpha2", "0.3.0a2"), ("v1.0.0-beta1", "1.0.0b1"),
    ("v1.0.0-rc2", "1.0.0rc2"), ("v1.2.3", "1.2.3"),
])
def test_tags_name_pep_440_versions(tag, version):
    assert check_release_tag.version_of_tag(tag) == version


@pytest.mark.parametrize("tag", ["0.3.0", "v0.3", "v0.3.0-gamma", "v0.3.0a0", "main"])
def test_other_refs_are_not_release_tags(tag):
    with pytest.raises(ValueError, match="not a release tag"):
        check_release_tag.version_of_tag(tag)


def test_the_tag_must_match_pyproject(capsys):
    version = check_release_tag.project_version()
    tag = "v" + version.replace("a", "-alpha", 1) if "a" in version else "v" + version
    assert check_release_tag.main([tag]) == 0
    assert check_release_tag.main(["v9.9.9"]) == 1
    assert "pyproject.toml says" in capsys.readouterr().err


def test_pushing_a_tag_never_publishes_on_its_own():
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text()
    publish = workflow[workflow.index("  publish:"):]
    assert "if: vars.PUBLISH_TO_PYPI == 'true'" in publish
    assert "id-token: write" in publish and "id-token" not in workflow[:workflow.index("  publish:")]
    assert "password" not in workflow and "secrets." not in workflow  # trusted publishing only
