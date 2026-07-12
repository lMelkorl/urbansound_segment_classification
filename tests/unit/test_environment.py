from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock

from urbansound_segment_task.edge_v2.utils.environment import (
    PACKAGE_DISTRIBUTIONS,
    SCHEMA_VERSION,
    collect_environment,
    collect_packages,
    collect_pytorch_runtime,
    serialize_environment,
    write_environment_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "audit_environment.py"
FIXED_TIME = datetime(2026, 7, 12, 12, 30, tzinfo=timezone.utc)


def _system_fixture() -> dict[str, object]:
    return {
        "os": "TestOS",
        "os_version": "1.2.3",
        "architecture": "test64",
        "cpu_name": "Test CPU",
        "physical_cores": 4,
        "logical_cores": 8,
        "total_ram_bytes": 16_000_000_000,
    }


def _python_fixture() -> dict[str, object]:
    return {
        "version": "3.12.0",
        "implementation": "CPython",
        "executable_name": "python3",
    }


def _packages_fixture() -> dict[str, dict[str, object]]:
    return {
        name: {"installed": name == "torch", "version": "2.0.0" if name == "torch" else None}
        for name, _ in PACKAGE_DISTRIBUTIONS
    }


def _git_fixture(_cwd: Optional[Path]) -> dict[str, object]:
    return {
        "available": True,
        "repository": True,
        "commit": "a" * 40,
        "branch": "feat/test",
        "clean": True,
    }


def _document(**overrides: object) -> dict[str, object]:
    kwargs = {
        "now": FIXED_TIME,
        "environ": {"OMP_NUM_THREADS": "4"},
        "system_collector": _system_fixture,
        "python_collector": _python_fixture,
        "package_collector": _packages_fixture,
        "pytorch_collector": lambda package: {
            "installed": package["installed"],
            "version": package["version"],
            "importable": True,
            "mps_available": False,
            "cuda_available": False,
        },
        "git_collector": _git_fixture,
    }
    kwargs.update(overrides)
    return collect_environment(**kwargs)


class EnvironmentCollectorTests(unittest.TestCase):
    def test_schema_version_and_required_fields(self) -> None:
        document = _document()

        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(document["created_at_utc"], "2026-07-12T12:30:00Z")
        for field in ("system", "python", "thread_environment", "packages", "pytorch", "git"):
            self.assertIn(field, document)
        self.assertEqual(set(document["packages"]), {name for name, _ in PACKAGE_DISTRIBUTIONS})

    def test_missing_optional_package_does_not_raise(self) -> None:
        def missing(_distribution: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        packages = collect_packages(version_lookup=missing)

        self.assertTrue(all(value == {"installed": False, "version": None} for value in packages.values()))
        runtime = collect_pytorch_runtime(packages["torch"], import_module=mock.Mock())
        self.assertFalse(runtime["installed"])
        self.assertFalse(runtime["importable"])

    def test_serialization_is_deterministic_for_same_inputs(self) -> None:
        first = serialize_environment(_document(), pretty=False)
        second = serialize_environment(_document(), pretty=False)

        self.assertEqual(first, second)

    def test_sensitive_values_and_fields_are_not_serialized(self) -> None:
        secret_values = {
            "HOME": "/Users/private-user",
            "USER": "private-user",
            "HOSTNAME": "private-host",
            "AWS_SECRET_ACCESS_KEY": "secret-aws-value",
            "GITHUB_TOKEN": "secret-github-value",
            "OMP_NUM_THREADS": "secret-thread-value",
        }
        serialized = serialize_environment(_document(environ=secret_values))

        for forbidden in (
            "/Users/private-user",
            "private-user",
            "private-host",
            "secret-aws-value",
            "secret-github-value",
            "secret-thread-value",
            '"hostname"',
            '"username"',
            '"email"',
            '"home"',
        ):
            self.assertNotIn(forbidden.lower(), serialized.lower())
        self.assertIn("redacted_invalid_value", serialized)

    def test_help_smoke(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
        self.assertIn("--pretty", result.stdout)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "environment.json"
            write_environment_file(output, "first\n")

            with self.assertRaises(FileExistsError):
                write_environment_file(output, "second\n")

            self.assertEqual(output.read_text(encoding="utf-8"), "first\n")

    def test_json_output_can_be_parsed_again(self) -> None:
        parsed = json.loads(serialize_environment(_document(), pretty=True))

        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        self.assertEqual(parsed["system"]["cpu_name"], "Test CPU")

    def test_pytorch_runtime_is_mockable(self) -> None:
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        runtime = collect_pytorch_runtime(
            {"installed": True, "version": "2.0.0"},
            import_module=lambda _name: fake_torch,
        )

        self.assertTrue(runtime["importable"])
        self.assertTrue(runtime["mps_available"])
        self.assertFalse(runtime["cuda_available"])


if __name__ == "__main__":
    unittest.main()
