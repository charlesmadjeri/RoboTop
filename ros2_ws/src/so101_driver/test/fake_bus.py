import rustypot

MODEL_NUMBER, DEFINITION = rustypot.find_model("sts3215")
REGISTER_NAMES = {register.name for register in DEFINITION.registers()}
MOTOR_IDS = (1, 2, 3, 4, 5, 6)
REST_TICK = 2048
DEFAULTS = {
    "model_number": MODEL_NUMBER,
    "firmware_major_version": 3,
    "firmware_minor_version": 10,
    "phase": 0,
    "torque_enable": 0,
    "present_position": REST_TICK,
    "goal_position": REST_TICK,
    "homing_offset": 0,
    "min_position_limit": 0,
    "max_position_limit": 4095,
}


class FakeBus:
    def __init__(self, registers=None, silent=()):
        self.registers = {motor_id: dict(DEFAULTS) for motor_id in MOTOR_IDS}
        for motor_id, values in (registers or {}).items():
            self.registers[motor_id].update(values)
        self.silent = set(silent)
        self.closed = False

    def _motor(self, motor_id):
        if self.closed or motor_id in self.silent or motor_id not in self.registers:
            raise RuntimeError(f"no answer from motor {motor_id}")
        return self.registers[motor_id]

    @staticmethod
    def _known(name):
        if name not in REGISTER_NAMES:
            raise ValueError(f"unknown register {name}")
        return name

    def _store(self, motor_id, name, value):
        motor = self._motor(motor_id)
        motor[self._known(name)] = int(value)
        if name == "goal_position" and motor["torque_enable"]:
            motor["present_position"] = int(value)

    def read_register_with_error(self, motor_id, name, retries=0):
        return self._motor(motor_id).get(self._known(name), 0), 0

    def write_register_with_error(self, motor_id, name, value, retries=0):
        self._store(motor_id, name, value)
        return 0

    def sync_read_register_with_error(self, motor_ids, name, retries=0):
        return [(self._motor(motor_id).get(self._known(name), 0), 0) for motor_id in motor_ids]

    def sync_write_register(self, motor_ids, name, values, retries=0):
        for motor_id, value in zip(motor_ids, values):
            self._store(motor_id, name, value)

    def set_torque(self, motor_ids, enabled, retries=0):
        failed = {}
        for motor_id in motor_ids:
            try:
                self._store(motor_id, "torque_enable", int(enabled))
            except RuntimeError as error:
                failed[motor_id] = str(error)
        return failed

    def close(self):
        self.closed = True
