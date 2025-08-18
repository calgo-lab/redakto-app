import flair
import os
from pathlib import Path
import logging

def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    flair.cache_root = Path(os.path.join(*['/app', 'flair_cache_root']))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Environment setup complete.")