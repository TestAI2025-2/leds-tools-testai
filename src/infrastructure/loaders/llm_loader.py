import os
from crewai import LLM

class LLM_Loader:
    @staticmethod
    def load_from_params(
        model: str = None,
        temp: float = None,
        api_key: str = None
    ) -> LLM:
        """
        Carrega a entidade LLM a partir de parâmetros fornecidos ou das variáveis de ambiente.
        """
        model = model if model is not None else os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash-lite-001")
        if not model:
            model = "gemini/gemini-2.0-flash-lite-001"

        temp = temp if temp is not None else float(os.getenv("LLM_TEMPERATURE", 0.0))
        api_key = api_key or os.getenv("GROQ_API_KEY")

        try:
            temp = float(temp)
        except Exception:
            raise ValueError("Temperature must be a float.")

        return LLM(
            model=model,
            temperature=temp,
            api_key=api_key
        )