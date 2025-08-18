"""Logging utilities for the RAG system.

Provides a colorized console formatter and a convenience `set_logger` function that can stream logs
both to a file and to the console. Use this to get consistent, contextual logs across the app.
"""
import logging, datetime as dt
from colorama import Fore, Style

class ColorFormmater(logging.Formatter):

    """Custom formatter that colorizes log messages based on level and prefixes a timestamp.

    The color map is:
    - INFO -> green
    - WARNING -> yellow
    - ERROR -> red
    """

    def format(self, record):

        message = super().format(record)
        _time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # map color to log level
        color_map = {
            logging.INFO: Fore.LIGHTGREEN_EX,
            logging.WARNING: Fore.LIGHTYELLOW_EX,
            logging.ERROR: Fore.LIGHTRED_EX
        }

        color_format = color_map.get(record.levelno)
        message = f"{color_format}[{record.levelname}] {_time}{Style.RESET_ALL}: {message}"
        return message

def set_logger(
    logger_name: str = None,
    log_level: int = logging.INFO,
    to_file: bool = False,
    log_file_name: str = None,
    to_console: bool = False,
    custom_formatter: logging.Formatter = None
) -> logging.Logger:
    """Create and configure a logger with optional file and console handlers.

    Args:
        logger_name (str, optional): Name of the logger. Defaults to the module name.
        log_level (int, optional): Logging level from the `logging` module. Defaults to `logging.INFO`.
        to_file (bool, optional): Whether to write logs to a file. Defaults to False.
        log_file_name (str, optional): Path to the log file required if `to_file=True`.
        to_console (bool, optional): Whether to stream logs to the console. Defaults to False.
        custom_formatter (logging.Formatter, optional): If provided, used for console output. Otherwise a
            simple default formatter is used.

    Raises:
        ValueError: If neither `to_file` nor `to_console` is True, or if `to_file=True` but no file name is provided.

    Returns:
        logging.Logger: Configured logger instance ready for use.
    """

    if not to_file and not to_console:
        raise ValueError("Must provide wither `to_file` or `to_console` for where to stream logs.")
    
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(log_level)

    # prevent multiple handlers
    if not logger.hasHandlers():

        default_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # log to file
        if to_file:
            if not log_file_name:
                raise ValueError("Must provide `log_file_name`")
            
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setLevel(log_level)

            file_handler.setFormatter(default_formatter)
            logger.addHandler(file_handler)

        # log to terminal
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            console_formatter = default_formatter if not custom_formatter else custom_formatter("%(message)s")

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger