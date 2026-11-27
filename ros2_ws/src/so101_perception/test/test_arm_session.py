import numpy as np
import pytest

from so101_perception.arm_session import (
    ARRIVAL_TIMEOUT_S,
    COMMAND_RATE_HZ,
    FRAMES_PER_POSE,
    collect_samples,
    cube_pixel,
    travel_to,
)

POSES = np.array([[0.1, 0.0, 0.0, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0, 0.0], [0.5, 0.2, 0.4, 0.3, 0.0]])
SPEED = 0.4


class FakeRig:
    def __init__(self, start=None, moves=True, gripper_known=True):
        self.joints = None if start is None else np.array(start, dtype=float)
        self.moves = moves
        self.gripper_known = gripper_known
        self.commands = []
        self.waited = 0.0
        self.shown = []

    def joint_positions(self):
        return self.joints

    def command(self, joints):
        self.commands.append(np.array(joints))
        if self.moves:
            self.joints = np.array(joints, dtype=float)

    def gripper_pose(self):
        return np.eye(4) if self.gripper_known else None

    def wait(self, seconds):
        self.waited += seconds

    def frame(self):
        self.shown.append(len(self.commands))
        return np.zeros((4, 4, 3), np.uint8)


def pixel_per_pose(rig, found=lambda pose_number: True):
    def detect(frame):
        pose_number = len(set(rig.shown))
        return (100.0 + pose_number, 50.0) if found(pose_number) else None

    return detect


def test_every_step_of_a_move_stays_below_the_speed_and_the_move_ends_on_the_target():
    rig = FakeRig(start=np.zeros(5))
    travel_to(rig, POSES[1], SPEED)
    path = np.vstack([np.zeros(5), *rig.commands])
    assert np.abs(np.diff(path, axis=0)).max() <= SPEED / COMMAND_RATE_HZ + 1e-9
    assert np.allclose(path[-1], POSES[1])
    assert rig.waited == pytest.approx(len(rig.commands) / COMMAND_RATE_HZ)


def test_a_move_without_joint_states_stops_with_a_message_and_sends_nothing():
    rig = FakeRig(start=None)
    with pytest.raises(RuntimeError, match="no /joint_states"):
        travel_to(rig, POSES[0], SPEED)
    assert rig.commands == []


def test_an_arm_that_never_arrives_is_reported_instead_of_photographed():
    rig = FakeRig(start=np.zeros(5), moves=False)
    with pytest.raises(RuntimeError, match="did not reach"):
        collect_samples(rig, rig, lambda frame: (1.0, 1.0), POSES, SPEED, report=lambda line: None)
    assert rig.waited >= ARRIVAL_TIMEOUT_S
    assert rig.shown == []


def test_samples_keep_the_pose_and_the_pixel_of_each_detected_pose_in_order():
    rig = FakeRig(start=POSES[0])
    samples = collect_samples(rig, rig, pixel_per_pose(rig), POSES, SPEED, report=lambda line: None)
    assert samples.kept == [1, 2, 3]
    assert samples.skipped == []
    assert samples.pixels.shape == (3, 2) and samples.gripper_poses.shape == (3, 4, 4)


def test_a_pose_with_no_detection_is_skipped_and_named_in_the_report():
    rig = FakeRig(start=POSES[0])
    lines = []
    samples = collect_samples(rig, rig, pixel_per_pose(rig, found=lambda pose_number: pose_number != 2), POSES, SPEED, lines.append)
    assert samples.kept == [1, 3] and samples.skipped == [2]
    assert any("pose 2/3" in line and "skipped" in line for line in lines)


def test_a_pose_with_an_unknown_gripper_frame_is_skipped():
    rig = FakeRig(start=POSES[0], gripper_known=False)
    samples = collect_samples(rig, rig, lambda frame: (1.0, 1.0), POSES, SPEED, report=lambda line: None)
    assert samples.kept == [] and samples.skipped == [1, 2, 3]


def test_the_pixel_of_a_pose_is_the_median_of_its_frames_and_ignores_one_wrong_frame():
    pixels = iter([(100.0, 50.0), (300.0, 200.0), (102.0, 52.0)])
    rig = FakeRig(start=POSES[0])
    assert np.allclose(cube_pixel(rig, lambda frame: next(pixels)), [102.0, 52.0])


def test_one_detection_out_of_the_frames_is_not_enough():
    answers = iter([(100.0, 50.0)] + [None] * (FRAMES_PER_POSE - 1))
    assert cube_pixel(FakeRig(start=POSES[0]), lambda frame: next(answers)) is None
