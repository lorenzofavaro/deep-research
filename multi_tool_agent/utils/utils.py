import random
import string
import uuid


def valid_uuid(max_length: int = 16) -> str:
    """
    Generate a valid UUID-like string with custom length and character constraints.

    Args:
        max_length: Maximum length of the generated string (default: 16)

    Returns:
        A string that starts with a letter or underscore and contains only
        letters, digits, and underscores
    """
    allowed_chars = string.ascii_letters + string.digits + '_'
    first_char_choices = string.ascii_letters + '_'
    base = uuid.uuid4().hex
    transformed = ''.join(random.choice(allowed_chars) for _ in base)
    result = random.choice(first_char_choices) + transformed[1:]
    return result[:max_length]
