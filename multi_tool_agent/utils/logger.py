import logging
import os


def setup_logger(
    name: str = 'multi_tool_agent',
    level: str = 'INFO',
) -> logging.Logger:
    """
    Set up a logger with console output only.

    Args:
        name: Name of the logger
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if logger.handlers:
        return logger

    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'multi_tool_agent') -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


_log_level = os.getenv('LOG_LEVEL', 'INFO')
main_logger = setup_logger(
    name='multi_tool_agent',
    level=_log_level,
)

# Configure ADK framework logging to reduce verbosity
google_adk_logger = logging.getLogger('google_adk')
google_adk_logger.setLevel(logging.WARNING)

if _log_level.upper() != 'DEBUG':
    logging.getLogger('google_adk.google.adk.models.google_llm').setLevel(
        logging.WARNING,
    )
    logging.getLogger('google_adk.google.adk.agents').setLevel(logging.WARNING)
    logging.getLogger('google_adk.google.adk.flows').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)

    logging.getLogger('google.adk').setLevel(logging.WARNING)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    )
