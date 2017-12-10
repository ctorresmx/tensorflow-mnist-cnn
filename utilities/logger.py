import logging
import logging.config

LOGGING = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)-15s %(levelname)-8s %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
        }
    }
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger()
