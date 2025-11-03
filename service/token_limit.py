def get_token_limit(model_id: str):
    """
    Get the token limit for a given model ID.
    
    Args:
        model_id (str): The model identifier
        
    Returns:
        int or str: Token limit for the model, or error message if not found
    """
    limits_by_model_id = {
        "gpt-5": 400000,
        "gpt-5-nano": 400000,
        "gpt-4.1": 1000000,
        "gemini-2.5-flash": 1000000,
        "gemini-2.5-pro": 1000000,
        "qwen3-235b-a22b-thinking-2507": 128000,
        "qwen3-235b-a22b-instruct-2507": 128000,
        "kimi-k2-0905-preview": 260000,
        "us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
        "anthropic/claude-sonnet-4.5": 200000,
        "@preset/claude-4-5": 200000,
        "glm-4.6": 200000,
        "glm-4.5": 128000,
        "deepseek-chat": 128000,  # deepseek v3.1 terminus
        "deepseek-reasoner": 128000,  # deepseek v3.1 terminus thinking mode
        "alibaba/tongyi-deepresearch-30b-a3b": 128000,
        "minimax/minimax-m2:free": 200000,
        "grok-4-0709": 256000,
    }
    
    key = (model_id or "").lower()
    return limits_by_model_id.get(key, f"Token limit not mapped for model id: {model_id}")