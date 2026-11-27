import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def scene():
    from perception_bench.scene import Scene

    try:
        built = Scene()
    except Exception as error:
        pytest.skip(f"no offscreen OpenGL: {error}")
    yield built
    built.close()
