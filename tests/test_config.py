from investigator.config import Settings


def test_chat_models_map_to_gemini_35_flash():
    settings = Settings(GEMINI_MODEL="gemini-2.5-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"
    settings = Settings(GEMINI_MODEL="gemini-3.6-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"
    settings = Settings(GEMINI_MODEL="gemini-3.5-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"
