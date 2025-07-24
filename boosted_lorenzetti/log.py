import logging
import logging.config

DEFAULT_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)s - %(message)s'


def set_logger(
    name: str = 'root',
    format: str = DEFAULT_FORMAT,
    level: int = logging.INFO,
    log_file: str | None = None,
):
    """Set up the logger for the application."""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': format
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default'
            }
        },
        'loggers': {
            name: {
                'handlers': ['console'],
                'level': level,
                'propagate': True
            }
        }
    }
    handlers = ['console']
    if log_file:
        handlers.append('file')
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'filename': log_file,
            'formatter': 'default'
        }
    logging.config.dictConfig(config)
    return logging.getLogger(name)
