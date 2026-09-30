"""Tests du chargement du modèle local (LocalLLMProvider), sans GPU ni poids :
torch et transformers sont remplacés par des doublures."""
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from backend.config import settings
from backend.llm_client import LLMError, LocalLLMProvider

CPU_DISPATCH = "Some modules are dispatched on the CPU or the disk."


def _load_local_provider(load_model):
    """Instancie LocalLLMProvider ; `load_model(name)` remplace
    AutoModelForCausalLM.from_pretrained. Retourne le nom du modèle chargé."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    transformers = MagicMock()
    transformers.AutoModelForCausalLM.from_pretrained.side_effect = (
        lambda name, **kwargs: load_model(name)
    )
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
         patch.object(settings, "LOCAL_LLM_MODEL", "primary-7b"), \
         patch.object(settings, "LOCAL_LLM_FALLBACK_MODEL", "fallback-3b"), \
         patch.object(LocalLLMProvider, "_model", None), \
         patch.object(LocalLLMProvider, "_tokenizer", None), \
         patch.object(LocalLLMProvider, "_model_name", None):
        LocalLLMProvider()
        return LocalLLMProvider._model_name


def _llm_client_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "backend.llm_client"]


def test_fallback_logs_which_model_failed_why_and_which_was_loaded(caplog):
    def load(name):
        if name == "primary-7b":
            raise ValueError(CPU_DISPATCH)
        return MagicMock()

    with caplog.at_level(logging.WARNING, logger="backend.llm_client"):
        loaded = _load_local_provider(load)

    assert loaded == "fallback-3b"
    assert _llm_client_warnings(caplog) == [
        "Modèle local de repli chargé : fallback-3b, à la place de "
        f"primary-7b (ValueError: {CPU_DISPATCH})"
    ]


def test_error_lists_every_failure_when_no_local_model_loads():
    def load(name):
        if name == "primary-7b":
            raise ValueError(CPU_DISPATCH)
        raise OSError("fallback-3b is not a local folder")

    with pytest.raises(LLMError) as exc:
        _load_local_provider(load)

    message = str(exc.value)
    assert f"primary-7b (ValueError: {CPU_DISPATCH})" in message
    assert "fallback-3b (OSError: fallback-3b is not a local folder)" in message


def test_no_warning_when_primary_model_loads(caplog):
    with caplog.at_level(logging.WARNING, logger="backend.llm_client"):
        loaded = _load_local_provider(lambda name: MagicMock())

    assert loaded == "primary-7b"
    assert _llm_client_warnings(caplog) == []
