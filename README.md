# Tek5: SO-ARM101 pick & place

Simulation, IK, perception and pick & place on the SO-ARM101 arm, with ROS 2 Jazzy and MuJoCo.
Project spec: [`docs/SUJET.md`](docs/SUJET.md).

## Prerequisites

Install [Pixi](https://pixi.sh) 0.84 or newer (already installed: `pixi self-update`), then open a new terminal.

### macOS, Linux

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Windows

Choose native or WSL2:

| | Native (PowerShell or Command Prompt) | WSL2 (Ubuntu) |
|---|---|---|
| RViz | GPU | CPU, slower |
| Setup | Pixi | `wsl --install -d Ubuntu` (admin), then as Linux |
| Python edits | Need `pixi run build` | Taken into account directly |
| Real arm | `COM` port | USB attached with [usbipd-win](https://learn.microsoft.com/windows/wsl/connect-usb) |

Native, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

WSL2: in Ubuntu, follow [macOS, Linux](#macos-linux).

## Quickstart

```bash
git clone https://github.com/charlesmadjeri/RoboTop.git
cd RoboTop
pixi install
pixi run build
pixi run launch --backend sim
```

Windows: clone in a path without spaces (e.g. `C:\RoboTop`). WSL2: clone in `~/`, not in `/mnt/c/`.

In a second terminal: `pixi run rviz2`.

## Commands

| Command | |
|---|---|
| `pixi run build` | Build `ros2_ws` |
| `pixi run launch --backend sim` | Build and launch the stack in simulation (default backend) |
| `pixi run launch --backend real --port <port>` | Same stack on the real arm |
| `pixi run rviz2` | RViz with `ros2_ws/so101.rviz` |
| `pixi shell` | Shell with ROS 2 and the workspace sourced |
| `pixi run ros2 <args>` | Any ROS 2 command |
| `pixi run doctor` | Check the setup, with the fix for each problem |
| `pixi run test` | Unit tests (`so101_driver` and the functional test helpers) |
| `pixi run functional-test` | Start the simulated stack and RViz headless, move the arm, save screenshots in `artifacts/` |
| `pixi run clean` | Delete `ros2_ws/build install log` |
| `pixi add <pkg>`, `pixi add --pypi <pkg>` | Add a dependency, commit `pixi.toml` and `pixi.lock` |

`launch` runs `ros2_ws/src/so101_bringup/launch/so101.launch.py`: add your nodes' launch files there.

## RViz

`pixi run rviz2` shows a grid, the robot model (`/robot_description`) and the simulation camera (`/external_cam/image_raw`). The model is preloaded in `ros2_ws/so101.rviz`, there is nothing to add by hand.

The arm stays "in pieces" until `driver_node` publishes `/joint_states` (see [docs/DRIVER.md](docs/DRIVER.md)).

## Real arm

In a `pixi shell`, use `lerobot-find-port`, `lerobot-calibrate` and `lerobot-find-cameras` as in the [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/main/en/so101) (`--robot.type=so101_follower`). Then copy `.env.example` to `.env`, fill in the port, the calibration file and the camera, and:

```bash
export SO101_CALIBRATION_FILE=<calibration file>     # PowerShell: $env:SO101_CALIBRATION_FILE = "<file>"
pixi run doctor                                      # checks them
pixi run launch --backend real --port <port>         # or set SO101_PORT
pixi run launch --backend real --port <port> --camera-index <n>   # USB camera, or set SO101_CAMERA
```

`--camera-index <n>` picks the USB camera of the arm (`-1` disables it). List the indexes with `lerobot-find-cameras`.

- **Linux**: `sudo usermod -aG dialout $USER`, then log out and back in. Use the port's stable name from `ls -l /dev/serial/by-id/`: `/dev/ttyACM0` can change when you plug the arms in another order. Another way to find the port, watch it appear when you plug the USB cable:
  `sudo udevadm monitor --subsystem-match=tty --property | grep --line-buffered -E "(DEVNAME=|ACTION=)"`
- **WSL2**: attach the USB device with [usbipd-win](https://learn.microsoft.com/windows/wsl/connect-usb).

## CI

`.github/workflows/ci.yml` runs on every pull request and on pushes to `dev`. The `unit` job installs the pixi environment as locked in `pixi.lock`, runs `pixi run build`, then `pixi run test`. Run the same two commands locally before you push.

The `functional` job runs `pixi run functional-test` under Xvfb with software rendering (`MUJOCO_GL=egl`). It checks `/joint_states` and `/external_cam/image_raw`, takes a screenshot of RViz and a sim camera frame, commands `shoulder_pan` to 0.5 rad, takes a second pair, and fails when the joint did not move or an image did not change. The screenshots, frames and logs are the `functional-test` artifact of the run. Locally it needs `Xvfb` and ImageMagick (`import`); `--target <rad>` changes the move.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ros2 topic list` does not see your nodes | Start each terminal with `pixi run` or `pixi shell` |
| Windows: odd `colcon` or `ros2` errors | Move the repo to a short path without spaces, e.g. `C:\RoboTop` |
| Windows: Python edit not taken into account | `pixi run build` (no symlinks on Windows) |
| Windows: firewall popup for `python.exe` | Cancel: ROS 2 only talks on your machine |
| WSL2: RViz is slow | CPU by default. GPU: `GALLIUM_DRIVER=d3d12 pixi run rviz2` (drop it if the view is empty) |
