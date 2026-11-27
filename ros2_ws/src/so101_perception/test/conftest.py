import pytest
from sim_arm import SimArm


@pytest.fixture(scope="session")
def arm():
    arm = SimArm()
    yield arm
    arm.close()
