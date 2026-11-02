from src.generation.prompts import IN_PDF_PROMPT, PERSONAS, with_persona


def test_default_persona_leaves_prompt_unchanged():
    assert with_persona(IN_PDF_PROMPT, "default") == IN_PDF_PROMPT
    assert with_persona(IN_PDF_PROMPT, None) == IN_PDF_PROMPT
    assert with_persona(IN_PDF_PROMPT, "unknown-persona") == IN_PDF_PROMPT


def test_every_persona_keeps_the_grounding_rules():
    for name in PERSONAS:
        prompt = with_persona(IN_PDF_PROMPT, name)
        assert prompt.startswith(IN_PDF_PROMPT)           # rules first, style appended after
        if name != "default":
            assert "never the citations" in prompt
            assert PERSONAS[name] in prompt
