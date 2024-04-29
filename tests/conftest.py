import platform
import shutil

import pytest

from bpm_ai_inference.util.hf import hf_home


@pytest.fixture(autouse=True, scope="module")
def cleanup():
    yield
    if platform.system() != "Darwin":
        cache_dir = hf_home()
        shutil.rmtree(cache_dir)