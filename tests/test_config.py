from investigator.config import Settings


def test_chat_models_map_to_gemini_35_flash():
    settings = Settings(GEMINI_MODEL="gemini-2.5-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"
    settings = Settings(GEMINI_MODEL="gemini-3.6-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"
    settings = Settings(GEMINI_MODEL="gemini-3.5-flash")
    assert settings.resolved_gemini_model == "gemini-3.5-flash"


def test_openrouter_and_gemini_embedding_models():
    settings = Settings(
        OPENROUTER_MODEL="openrouter/free",
        EMBEDDING_MODEL="models/text-embedding-004",
    )
    assert settings.resolved_openrouter_model == "openrouter/free"
    assert settings.resolved_embedding_model == "models/gemini-embedding-001"

    legacy = Settings(OPENROUTER_MODEL="gemini-2.5-flash")
    assert legacy.resolved_openrouter_model == "openrouter/free"
