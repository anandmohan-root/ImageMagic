"""The app must never touch the network."""
import ast
import socket
from pathlib import Path

import pytest

from bgreplacer import pipeline

PKG = Path(pipeline.__file__).parent
NETWORK_MODULES = {"socket", "urllib", "http", "requests", "httpx", "aiohttp",
                   "ftplib", "smtplib", "telnetlib", "websocket", "ssl"}


@pytest.fixture
def no_sockets(monkeypatch):
    """Fail the test if any socket is created or any DNS lookup happens."""
    opened = []

    class GuardSocket(socket.socket):
        def __init__(self, *a, **k):
            opened.append(a)
            raise AssertionError(f"socket opened during run: {a}")

    def guard(*a, **k):
        opened.append(a)
        raise AssertionError(f"network call during run: {a}")

    monkeypatch.setattr(socket, "socket", GuardSocket)
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "socketpair"):
        monkeypatch.setattr(socket, name, guard)
    yield opened
    assert opened == [], f"network activity: {opened}"


def test_guard_actually_trips(no_sockets):
    with pytest.raises(AssertionError):
        socket.create_connection(("example.com", 80))
    no_sockets.clear()


def test_full_run_opens_no_socket(no_sockets, tiny_model, jpegs):
    main, bg, out = jpegs
    seg = pipeline.Segmenter.load_default(tiny_model)
    assert pipeline.replace_background(main, bg, out, seg).exists()


@pytest.mark.skipif(not pipeline.find_models(), reason="bundled model not downloaded")
def test_real_model_run_opens_no_socket(no_sockets, jpegs):
    main, bg, out = jpegs
    seg = pipeline.Segmenter.load_default()
    assert pipeline.replace_background(main, bg, out, seg).exists()


def test_package_has_no_network_imports():
    for py in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in NETWORK_MODULES, f"{py.name} imports {n}"
