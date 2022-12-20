import importlib.metadata

from transformers import logging

logging.set_verbosity_error()

try:
    __version__ = importlib.metadata.version("glip-object-detection")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""
