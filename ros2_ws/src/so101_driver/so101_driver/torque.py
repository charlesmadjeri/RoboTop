def set_torque(robot, enabled: bool) -> None:
    bus = robot.bus
    bus.disable_torque()
    if not enabled:
        return
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
    bus.enable_torque()
