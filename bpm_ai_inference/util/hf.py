import os


def hf_home():
    return os.getenv('HF_HOME', os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))