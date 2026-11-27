import numpy as np
import pytest
import rclpy
from arm_scene import cube_pixels, gripper_poses, noisy_poses
from cube_scene import K, ROBOT_FROM_CAMERA

from so101_perception import calibrate_from_arm
from so101_perception.arm_poses import PRESET_POSES
from so101_perception.arm_session import Samples
from so101_perception.calibrate_from_arm import camera_matrix, evaluate, main, parse_args, refusal, summary


def samples_of(poses, pixels):
    return Samples(pixels, poses, list(range(1, len(poses) + 1)), [])


@pytest.mark.parametrize("speed", ["0", "-1", "1.5", "fast"])
def test_a_speed_that_is_not_a_slow_positive_one_is_refused_before_anything_moves(speed, capsys):
    with pytest.raises(SystemExit):
        parse_args(["--speed", speed])
    assert "--speed" in capsys.readouterr().err


def test_consistent_poses_are_accepted_and_the_held_out_error_is_in_centimeters():
    poses = gripper_poses()
    evaluation = evaluate(samples_of(poses, cube_pixels(ROBOT_FROM_CAMERA, poses)), K)
    assert evaluation.held_out_cm.max() < 0.1
    assert refusal(evaluation, max_error_cm=1.0) is None


def test_arm_positions_that_are_off_by_centimeters_are_refused_by_their_held_out_error():
    poses = gripper_poses()
    evaluation = evaluate(samples_of(noisy_poses(poses, 0.015, seed=0), cube_pixels(ROBOT_FROM_CAMERA, poses)), K)
    message = refusal(evaluation, max_error_cm=1.0)
    assert "median held-out error" in message and "above 1.0 cm" in message


def test_arm_positions_that_are_off_by_three_centimeters_fit_no_camera_pose_and_say_so():
    poses = gripper_poses()
    with pytest.raises(ValueError, match="do not fit any camera pose"):
        evaluate(samples_of(noisy_poses(poses, 0.03), cube_pixels(ROBOT_FROM_CAMERA, poses)), K)


def test_a_result_that_cannot_be_checked_on_held_out_poses_is_refused():
    poses = gripper_poses(count=8)
    evaluation = evaluate(samples_of(poses, cube_pixels(ROBOT_FROM_CAMERA, poses)), K)
    assert evaluation.held_out_cm is None
    assert "too few poses" in refusal(evaluation, max_error_cm=1.0)


def test_the_summary_names_the_dropped_and_the_skipped_poses():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses)
    pixels[4] += [150.0, -90.0]
    samples = Samples(pixels, poses, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13], [5])
    text = summary(samples, evaluate(samples, K))
    assert "pose 6: " in text and "dropped as a wrong detection" in text
    assert "skipped, not detected: [5]" in text
    assert "held-out error: median" in text


def test_the_field_of_view_wins_then_the_camera_info_then_a_default_of_75_degrees(capsys):
    frame = np.zeros((720, 1280, 3), np.uint8)
    from_topic = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    assert camera_matrix(parse_args(["--fov-deg", "90"]), from_topic, frame)[0, 0] == pytest.approx(640.0)
    assert np.array_equal(camera_matrix(parse_args([]), from_topic, frame), from_topic)
    assert camera_matrix(parse_args([]), None, frame)[0, 0] == pytest.approx(640.0 / np.tan(np.radians(37.5)))
    assert "75" in capsys.readouterr().out


def test_a_dry_run_prints_the_plan_and_never_starts_ros_or_moves_the_arm(capsys, monkeypatch):
    monkeypatch.setattr(calibrate_from_arm, "run_on_robot", lambda *args: pytest.fail("the arm must not move"))
    main(["--dry-run"])
    text = capsys.readouterr().out
    assert f"{len(PRESET_POSES)} poses" in text and "longest move" in text
    assert not rclpy.ok()


def test_a_real_run_hands_the_poses_to_the_robot_code(monkeypatch):
    seen = []
    monkeypatch.setattr(calibrate_from_arm, "run_on_robot", lambda args, poses: seen.append(poses))
    main([])
    assert np.array_equal(seen[0], PRESET_POSES)


def test_a_poses_file_beyond_the_joint_limits_stops_before_anything_else(tmp_path, monkeypatch):
    poses = tmp_path / "poses.json"
    poses.write_text("[[3.0, 0, 0, 0, 0]]")
    monkeypatch.setattr(calibrate_from_arm, "run_on_robot", lambda *args: pytest.fail("the arm must not move"))
    with pytest.raises(SystemExit, match="pose 1 puts shoulder_pan too close to its limit"):
        main(["--poses", str(poses)])
