import random
import string
import uuid


def valid_uuid(max_length=16):
    allowed_chars = string.ascii_letters + string.digits + '_'
    first_char_choices = string.ascii_letters + '_'
    base = uuid.uuid4().hex
    transformed = ''.join(random.choice(allowed_chars) for _ in base)
    result = random.choice(first_char_choices) + transformed[1:]
    return result[:max_length]
