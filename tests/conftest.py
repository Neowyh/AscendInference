from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


class LocalTempPathFactory:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def getbasetemp(self) -> Path:
        return self._base_dir

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        safe_name = basename.replace("\\", "_").replace("/", "_")
        if not numbered:
            target = self._base_dir / safe_name
            target.mkdir(parents=True, exist_ok=True)
            return target

        index = 0
        while True:
            target = self._base_dir / f"{safe_name}-{index}"
            try:
                target.mkdir(parents=True, exist_ok=False)
                return target
            except FileExistsError:
                index += 1


@pytest.fixture(scope="session")
def tmp_path_factory() -> LocalTempPathFactory:
    workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    base_dir = Path(tempfile.mkdtemp(prefix="pytest-", dir=workspace_tmp))
    factory = LocalTempPathFactory(base_dir)
    yield factory
    shutil.rmtree(base_dir, ignore_errors=True)


@pytest.fixture
def tmp_path(tmp_path_factory: LocalTempPathFactory, request: pytest.FixtureRequest) -> Path:
    return tmp_path_factory.mktemp(request.node.name, numbered=True)
