import os
import pytest
from src.infrastructure.loaders.llm_loader import LLM_Loader
from crewai import LLM

def test_llm_loader_usa_ambiente(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mdl-env")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("GEMINI_API_KEY", "key-env")
    inst = LLM_Loader.load_from_params()
    assert isinstance(inst, LLM)
    assert inst.model == "mdl-env"
    assert abs(inst.temperature - 0.25) < 1e-6
    assert inst.api_key == "key-env"

def test_llm_loader_parametros_explicitamente():
    inst = LLM_Loader.load_from_params(model="mdl1", temp=0.5, api_key="key1")
    assert inst.model == "mdl1"
    assert inst.temperature == 0.5
    assert inst.api_key == "key1"

def test_llm_loader_default_sem_ambiente(tmp_path, monkeypatch):
    # remove variáveis de ambiente
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    inst = LLM_Loader.load_from_params()
    assert inst.model == "gemini/gemini-2.0-flash-lite-001"
    assert inst.temperature == 0.0
    assert inst.api_key is None

def test_llm_loader_temperatura_invalida_str(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "abc")
    with pytest.raises(ValueError):
        LLM_Loader.load_from_params()

def test_llm_loader_temperatura_int():
    inst = LLM_Loader.load_from_params(temp=1)
    assert inst.temperature == 1.0

def test_llm_loader_model_empty_str(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    # modelo vazio deve usar default
    inst = LLM_Loader.load_from_params()
    assert inst.model == "gemini/gemini-2.0-flash-lite-001"

def test_llm_loader_api_key_vazia(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    inst = LLM_Loader.load_from_params()
    assert inst.api_key == ""

def test_llm_loader_negative_temperature():
    inst = LLM_Loader.load_from_params(temp=-0.1, model="m2", api_key="k2")
    assert inst.temperature == -0.1
