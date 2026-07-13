"""Tests for zea.log: level functions, location attribution, warning dedup/suppression,
file-logger mirroring, and progress-bar coordination.
"""

import gc
import io
import logging

import pytest
from tqdm import tqdm

import zea
from zea import log

_ZEA_PACKAGE_DIR = zea.__file__.rsplit("/", 1)[0]


@pytest.fixture
def attach_caplog(caplog):
    """Attaches pytest's caplog handler directly to zea's logger.

    ``zea.log``'s logger has ``propagate = False``, so the default caplog
    mechanism (which relies on records bubbling up to the root logger) can't
    see it. Attaching the handler directly sidesteps that.
    """
    caplog.set_level(logging.DEBUG)
    log.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        log.logger.removeHandler(caplog.handler)


@pytest.fixture
def file_logger_capture():
    """Points ``log.file_logger`` at an in-memory logger for the duration of a test.

    Bypasses ``enable_file_logging``/``configure_file_logger`` (which touch real
    files and a module-global directory) so tests stay hermetic.
    """
    original = log.file_logger
    stream = io.StringIO()
    fl = logging.getLogger("test_file_logger_capture")
    fl.handlers.clear()
    fl.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    fl.addHandler(handler)
    fl.propagate = False
    log.file_logger = fl
    try:
        yield stream
    finally:
        log.file_logger = original


@pytest.fixture(autouse=True)
def _isolate_warned_locations():
    """Prevents ``warning_once`` dedup state from leaking between tests."""
    saved = set(log._warned_locations)
    log._warned_locations.clear()
    yield
    log._warned_locations.clear()
    log._warned_locations.update(saved)


def _compile_in_fake_zea_module(src, fake_basename):
    """Executes ``src`` reporting its call frames as living inside the zea package.

    Lets us simulate a message genuinely originating from deep inside a zea
    internal helper, to test the "skip to the user's frame" logic, without
    touching any real source file.
    """
    fake_path = f"{_ZEA_PACKAGE_DIR}/{fake_basename}"
    namespace = {}
    exec(compile(src, fake_path, "exec"), namespace)
    return namespace, fake_path


# --------------------------------------------------------------------------- #
# Basic level functions
# --------------------------------------------------------------------------- #

LEVEL_FUNCS = [
    ("debug", logging.DEBUG),
    ("info", logging.INFO),
    ("warning", logging.WARNING),
    ("error", logging.ERROR),
    ("critical", logging.CRITICAL),
    ("deprecated", log.DEPRECATED_LEVEL_NUM),
]


@pytest.mark.parametrize("func_name,level", LEVEL_FUNCS)
def test_level_functions_log_at_correct_level(attach_caplog, func_name, level):
    func = getattr(log, func_name)
    result = func(f"hello from {func_name}")
    assert attach_caplog.records[-1].levelno == level
    assert f"hello from {func_name}" in attach_caplog.records[-1].message
    assert result == f"hello from {func_name}"


def test_success_logs_info_in_green_and_returns_plain_message(attach_caplog):
    result = log.success("all good")
    assert result == "all good"
    assert attach_caplog.records[-1].levelno == logging.INFO
    assert "all good" in attach_caplog.records[-1].message


# --------------------------------------------------------------------------- #
# location / raw_location
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("func_name", ["debug", "info", "warning", "error", "critical"])
def test_location_direct_call_points_to_caller(attach_caplog, func_name):
    func = getattr(log, func_name)
    result = func("msg", location=True)
    assert result.startswith(f"{__file__}:")
    assert result.endswith(": msg")


def test_location_default_skips_internal_zea_frames(attach_caplog):
    namespace, fake_path = _compile_in_fake_zea_module(
        "from zea import log\n"
        "def internal_helper():\n"
        "    return log.warning('deep message', location=True)\n",
        "_fake_internal_module.py",
    )
    result = namespace["internal_helper"]()
    assert not result.startswith(fake_path)
    assert result.startswith(f"{__file__}:")
    assert result.endswith(": deep message")


def test_raw_location_shows_literal_zea_internal_frame(attach_caplog):
    namespace, fake_path = _compile_in_fake_zea_module(
        "from zea import log\n"
        "def internal_helper():\n"
        "    return log.warning('deep message', location=True, raw_location=True)\n",
        "_fake_internal_module.py",
    )
    result = namespace["internal_helper"]()
    assert result.startswith(f"{fake_path}:")


def test_caller_frame_falls_back_when_whole_stack_is_internal(monkeypatch):
    # A real all-internal call stack is unreachable through pytest (pytest's own
    # frames are always external to zea), so unit-test `_caller_frame` directly
    # against a synthetic stack instead.
    class _FakeFrame:
        def __init__(self, filename, lineno):
            self.filename = filename
            self.lineno = lineno

    fake_frames = [
        _FakeFrame(f"{_ZEA_PACKAGE_DIR}/models/gmm.py", 10),
        _FakeFrame(f"{_ZEA_PACKAGE_DIR}/utils.py", 20),
    ]
    monkeypatch.setattr(log.inspect, "stack", lambda: [None, *fake_frames])
    frame = log._caller_frame(skip_package=True)
    assert frame is fake_frames[0]


# --------------------------------------------------------------------------- #
# warning_once
# --------------------------------------------------------------------------- #


def test_warning_once_dedupes_by_call_site(attach_caplog):
    def call_a():
        log.warning_once("dedup A")

    def call_b():
        log.warning_once("dedup B")

    for _ in range(3):
        call_a()
    for _ in range(3):
        call_b()

    messages = [r.message for r in attach_caplog.records]
    assert messages.count("dedup A") == 1
    assert messages.count("dedup B") == 1


def test_warning_once_key_scopes_dedup_independent_of_call_site(attach_caplog):
    def call(key):
        for _ in range(3):
            log.warning_once("keyed message", key=key)

    call("a")
    call("b")

    messages = [r.message for r in attach_caplog.records if r.message == "keyed message"]
    assert len(messages) == 2


def test_warning_once_forwards_location(attach_caplog):
    result = log.warning_once("located once", location=True)
    # warning_once always returns the original bare message - the location
    # prefix only affects what actually gets printed/logged.
    assert result == "located once"
    assert attach_caplog.records[-1].message.startswith(f"{__file__}:")


# --------------------------------------------------------------------------- #
# suppress_warnings
# --------------------------------------------------------------------------- #


def test_suppress_warnings_only_suppresses_warning_family(attach_caplog):
    with log.suppress_warnings():
        log.warning("suppressed warning")
        log.warning_once("suppressed warning once")
        log.deprecated("suppressed deprecated")
        log.info("not suppressed info")
        log.debug("not suppressed debug")
        log.error("not suppressed error")
        log.critical("not suppressed critical")

    messages = {r.message for r in attach_caplog.records}
    assert "suppressed warning" not in messages
    assert "suppressed warning once" not in messages
    assert "suppressed deprecated" not in messages
    assert "not suppressed info" in messages
    assert "not suppressed debug" in messages
    assert "not suppressed error" in messages
    assert "not suppressed critical" in messages


def test_suppress_warnings_resets_after_context_exits(attach_caplog):
    with log.suppress_warnings():
        pass
    log.warning("visible again")
    assert any(r.message == "visible again" for r in attach_caplog.records)


# --------------------------------------------------------------------------- #
# file logger mirroring
# --------------------------------------------------------------------------- #


def test_file_logger_strips_ansi_for_every_level(file_logger_capture):
    for func_name in ["debug", "info", "warning", "error", "critical", "deprecated"]:
        getattr(log, func_name)(log.red(f"colored {func_name}"))
    content = file_logger_capture.getvalue()
    assert "\x1b[" not in content
    for func_name in ["debug", "info", "warning", "error", "critical", "deprecated"]:
        assert f"colored {func_name}" in content


def test_file_logger_mirrors_success_without_color(file_logger_capture):
    log.success("saved file")
    content = file_logger_capture.getvalue()
    assert "\x1b[" not in content
    assert "saved file" in content


# --------------------------------------------------------------------------- #
# progress-bar coordination
# --------------------------------------------------------------------------- #


class _FakeBar:
    def __init__(self):
        self.redraw_calls = 0

    def redraw(self):
        self.redraw_calls += 1


def test_register_and_unregister_progress():
    bar = _FakeBar()
    log.register_progress(bar)
    assert bar in log._active_progress
    log.unregister_progress(bar)
    assert bar not in log._active_progress


def test_active_progress_is_weakly_referenced():
    gc.collect()
    baseline = len(log._active_progress)

    bar = _FakeBar()
    log.register_progress(bar)
    assert len(log._active_progress) == baseline + 1

    del bar
    gc.collect()
    assert len(log._active_progress) == baseline


class _CapturingHandler:
    """Swaps in a `_ProgressAwareStreamHandler` writing to an in-memory buffer.

    `log.logger`'s real console handler binds its stream once at import time, so
    relying on it (via capsys/capfd) is unreliable under pytest. Attaching our own
    handler with a controlled stream sidesteps that - and any other handler
    (e.g. that real console handler, or another test's caplog handler) is
    temporarily removed so it can't double-count effects like bar redraws.
    """

    def __enter__(self):
        self.stream = io.StringIO()
        self.handler = log._ProgressAwareStreamHandler(stream=self.stream)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self._saved_handlers = list(log.logger.handlers)
        for handler in self._saved_handlers:
            log.logger.removeHandler(handler)
        log.logger.addHandler(self.handler)
        return self.stream

    def __exit__(self, *exc_info):
        log.logger.removeHandler(self.handler)
        for handler in self._saved_handlers:
            log.logger.addHandler(handler)


def test_handler_clears_line_and_redraws_registered_bars():
    gc.collect()
    assert not log._tqdm_cls._instances
    bar = _FakeBar()
    log.register_progress(bar)
    try:
        with _CapturingHandler() as stream:
            log.info("hello custom bar")
            out = stream.getvalue()
    finally:
        log.unregister_progress(bar)

    assert bar.redraw_calls == 1
    assert "hello custom bar" in out
    assert "\x1b[K" in out


def test_handler_prefers_tqdm_over_registered_bars_when_active():
    bar = _FakeBar()
    log.register_progress(bar)
    try:
        with _CapturingHandler() as stream:
            with tqdm(total=1, file=stream):
                log.info("hello via tqdm")
            out = stream.getvalue()
    finally:
        log.unregister_progress(bar)

    assert bar.redraw_calls == 0
    assert "hello via tqdm" in out


def test_handler_behaves_like_plain_stream_handler_when_idle():
    gc.collect()
    assert not log._tqdm_cls._instances
    with _CapturingHandler() as stream:
        log.info("plain message")
        out = stream.getvalue()
    assert out == "plain message\n"
