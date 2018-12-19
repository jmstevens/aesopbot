import json
import logging
import sys

logger = logging.getLogger(__name__)

def loadConfig():
    global cfg
    try:
        with open('src/models/config.json','r') as cfgFile:
            try:
                cfg = json.load(cfgFile)
                logger.info("config.json loaded")
                return True
            except json.decoder.JSONDecodeError as e:
                logger.critical("config.json is not formatted properly {0}".format(e))
                return False
    except FileNotFoundError as e:
        logger.critical("config.json file missing inside root folder {0}".format(e))
        return False

if loadConfig() is not True:
    logger.critical("Bot not started. Check config file.")
    sys.exit(0)
