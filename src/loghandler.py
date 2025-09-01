"""
Logging utilities for the RAG system.

This module provides a custom colorized console formatter (`ColorFormatter`) and a utility function
(`set_logger`) to configure logging with consistent formatting for both file and console output.
Logs are color-coded by level (INFO: green, WARNING: yellow, ERROR: red) and include timestamps.
"""
import logging
import datetime as dt
from colorama import Fore, Style

class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that colorizes console output based on log level.

    Adds a timestamp and color-codes log messages by level:
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red

    Attributes:
        color_map (dict): Mapping of log levels to colorama color codes.
    """
    def format(self, record):
        """
        Format a log record with color and timestamp.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with color and timestamp.
        """
        message = super().format(record)
        _time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    """
    Create and configure a logger with optional file and console handlers.

    Configures a logger with the specified name, level, and output destinations (file and/or console).
    Prevents duplicate handlers and supports a custom formatter for console output.

    Args:
        logger_name (str, optional): Name of the logger. Defaults to the module name if None.
        log_level (int, optional): Logging level (e.g., logging.INFO). Defaults to logging.INFO.
        to_file (bool, optional): Whether to write logs to a file. Defaults to False.
        log_file_name (str, optional): Path to the log file if `to_file` is True.
        to_console (bool, optional): Whether to stream logs to the console. Defaults to False.
        custom_formatter (logging.Formatter, optional): Custom formatter for console output. Defaults to a simple formatter.

    Returns:
        logging.Logger: A configured logger instance.

    Raises:
        ValueError: If neither `to_file` nor `to_console` is True, or if `to_file` is True but no `log_file_name` is provided.
    """
    if not to_file and not to_console:
        raise ValueError("Must provide either `to_file` or `to_console` for where to stream logs.")
    
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(log_level)

    # Prevent multiple handlers
    if not logger.hasHandlers():
        default_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Log to file
        if to_file:
            if not log_file_name:
                raise ValueError("Must provide `log_file_name`")
            
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(default_formatter)
            logger.addHandler(file_handler)

        # Log to terminal
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = default_formatter if not custom_formatter else custom_formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger