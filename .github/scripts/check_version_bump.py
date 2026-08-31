"""Require a SemVer bump when files under src/ change."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

# https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def parse_version(value: str) -> tuple[tuple[int, int, int], tuple[str, ...] | None]:
    match = SEMVER_RE.fullmatch(value)
    if not match:
        raise ValueError(f"{value!r} is not valid SemVer 2.0.0")

    prerelease = match.group("prerelease")
    return (
        (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
        ),
        tuple(prerelease.split(".")) if prerelease else None,
    )


def version_is_greater(current: str, previous: str) -> bool:
    current_core, current_pre = parse_version(current)
    previous_core, previous_pre = parse_version(previous)
    if current_core != previous_core:
        return current_core > previous_core
    if current_pre is None:
        return previous_pre is not None
    if previous_pre is None:
        return False

    for current_part, previous_part in zip(current_pre, previous_pre):
        if current_part == previous_part:
            continue
        if current_part.isdigit() and previous_part.isdigit():
            return int(current_part) > int(previous_part)
        if current_part.isdigit():
            return False
        if previous_part.isdigit():
            return True
        return current_part > previous_part
    return len(current_pre) > len(previous_pre)


def project_version(pyproject: str) -> str:
    return tomllib.loads(pyproject)["project"]["version"]


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_sha")
    parser.add_argument("head_sha")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    current = project_version(Path("pyproject.toml").read_text())
    parse_version(current)
    previous = project_version(git_output("show", f"{args.base_sha}:pyproject.toml"))
    parse_version(previous)
    source_files = git_output(
        "diff", "--name-only", f"{args.base_sha}...{args.head_sha}", "--", "src/"
    )
    test_files = git_output(
        "diff", "--name-only", f"{args.base_sha}...{args.head_sha}", "--", "tests/"
    )
    source_changed = bool(source_files)
    tests_changed = bool(test_files)
    version_changed = current != previous

    print(f"Base version: `{previous}`")
    print(f"Current version: `{current}`")
    print(f"Source changed: `{source_changed}`")
    print(f"Tests changed: `{tests_changed}`")
    print(f"Version changed: `{version_changed}`")
    if source_changed:
        print("A valid version bump was required because `src/` changed.")
    else:
        print("No source files changed; a version bump was not required.")

    if source_changed and not version_is_greater(current, previous):
        raise ValueError(
            "Files under src/ changed, so project.version must be greater than "
            f"the base version ({previous}); found {current}."
        )

    if args.github_output:
        args.github_output.write_text(
            f"source_changed={str(source_changed).lower()}\n"
            f"tests_changed={str(tests_changed).lower()}\n"
            f"version={current}\n"
            f"version_changed={str(version_changed).lower()}\n"
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
