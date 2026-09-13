"""Tests for the logging configuration convention."""

import logging

from core import logging_config


def _reset() -> None:
    """Undo module-level configuration state and any handlers it attached."""
    logging_config._configured = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


class TestConfigureLogging:
    def setup_method(self) -> None:
        _reset()

    def teardown_method(self) -> None:
        _reset()

    def test_sets_root_level_from_explicit_argument(self) -> None:
        logging_config.configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_defaults_to_info_when_env_var_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("FACTLAMA_LOG_LEVEL", raising=False)
        logging_config.configure_logging()
        assert logging.getLogger().level == logging.INFO

    def test_reads_level_from_environment_variable(self, monkeypatch) -> None:
        monkeypatch.setenv("FACTLAMA_LOG_LEVEL", "WARNING")
        logging_config.configure_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_is_idempotent_and_does_not_duplicate_handlers(self) -> None:
        logging_config.configure_logging(level="INFO")
        handler_count = len(logging.getLogger().handlers)

        logging_config.configure_logging(level="DEBUG")

        assert len(logging.getLogger().handlers) == handler_count
        # The second call was a no-op: level from the first call is retained.
        assert logging.getLogger().level == logging.INFO
