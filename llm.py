import config
from langchain_groq import ChatGroq


def get_llm(temperature=0.0):
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=temperature,
        api_key=config.GROQ_API_KEY,
        timeout=20,       # secondes d'attente maximum par requête
        max_retries=1,    # le client réessaie une fois ; notre boucle gère le reste
    )


def structured_llm(schema):
    """LLM qui renvoie directement un objet Pydantic valide."""
    return get_llm().with_structured_output(schema, method="function_calling")