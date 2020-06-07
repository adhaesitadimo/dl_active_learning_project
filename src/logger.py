import sys


def initialize_logger(file_path, name='biomed_ie'):
    #import flair # Logging issue of 0.4.0
    import logging
    import logging.handlers

    
#     logging.config.dictConfig({
#         'version': 1,
#         'disable_existing_loggers': True,
#         'formatters': {
#             'standard': {
#                 'format': '%(asctime)-15s %(message)s'
#             },
#         },
#         'handlers': {
#             'console': {
#                 'level': 'INFO',
#                 'class': 'logging.StreamHandler',
#                 'formatter': 'standard',
#                 'stream': 'ext://sys.stdout'
#             },
#         },
#         'loggers': {
#             'flair': {
#                 'handlers': ['console'],
#                 'level': 'INFO',
#                 'propagate': False
#             }
#         },
#         'root': {
#             'handlers': [],
#             'level': 'WARNING'
#         }
#     })

    logger = logging.getLogger(name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #logger.handlers = []

    fhandler = logging.handlers.TimedRotatingFileHandler(filename=file_path, when='midnight')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)
    
    return logger
