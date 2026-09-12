from investigator.config import Settings


def test_retired_chat_models_map_to_gemini_36():
    settings = Settings(GEMINI_MODEL="gemini-2.5-flash")
    assert settings.resolved_gemini_model == "gemini-3.6-flash"
    settings = Settings(GEMINI_MODEL="gemini-3.5-flash-lite")
    assert settings.resolved_gemini_model == "gemini-3.6-flash"
