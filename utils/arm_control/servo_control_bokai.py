from dynamixel_sdk import *
import time


class DynamixelMotorBase:
    """
    Base class for Dynamixel motors with common functionality.
    """

    # Common control table addresses
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_MIN_POSITION_LIMIT = 52
    ADDR_MAX_POSITION_LIMIT = 48
    PROTOCOL_VERSION = 2.0

    def __init__(self, motor_id, port_handler, packet_handler):
        """
        Initialize base motor properties.

        Args:
            motor_id (int): ID of the motor
            port_handler: PortHandler instance
            packet_handler: PacketHandler instance
        """
        self.id = motor_id
        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.min_position = None
        self.max_position = None

    def enable_torque(self):
        """Enable torque for the motor."""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.id, self.ADDR_TORQUE_ENABLE, 1
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to enable torque on motor {self.id}")

    def disable_torque(self):
        """Disable torque for the motor."""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.id, self.ADDR_TORQUE_ENABLE, 0
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to disable torque on motor {self.id}")

    def read_position_limits(self):
        """Read position limits from the motor."""
        # Read min position limit
        min_pos, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.id, self.ADDR_MIN_POSITION_LIMIT
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to read min position limit for motor {self.id}")

        # Read max position limit
        max_pos, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.id, self.ADDR_MAX_POSITION_LIMIT
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to read max position limit for motor {self.id}")

        self.min_position = min_pos
        self.max_position = max_pos
        return min_pos, max_pos

    def set_position_limits(self, min_pos, max_pos):
        """
        Set custom position limits for the motor.

        Args:
            min_pos (int): Minimum position limit
            max_pos (int): Maximum position limit
        """
        self.min_position = min_pos
        self.max_position = max_pos
        self.range = max_pos - min_pos
        self.write_position_limits()

    def write_position_limits(self):
        """Write position limits to the motor."""
        if self.min_position is None or self.max_position is None:
            raise ValueError("Position limits not set")

        # Store current torque state
        torque_enabled = False
        value, result, error = self.packet_handler.read1ByteTxRx(
            self.port_handler, self.id, self.ADDR_TORQUE_ENABLE
        )
        if result == COMM_SUCCESS and error == 0:
            torque_enabled = value == 1

        # Disable torque if it was enabled
        if torque_enabled:
            self.disable_torque()

        try:
            # Write max position limit first (recommended order)
            result, error = self.packet_handler.write4ByteTxRx(
                self.port_handler,
                self.id,
                self.ADDR_MAX_POSITION_LIMIT,
                self.max_position,
            )
            if result != COMM_SUCCESS or error != 0:
                raise Exception(
                    f"Failed to write max position limit for motor {self.id}"
                )

            # Write min position limit
            result, error = self.packet_handler.write4ByteTxRx(
                self.port_handler,
                self.id,
                self.ADDR_MIN_POSITION_LIMIT,
                self.min_position,
            )
            if result != COMM_SUCCESS or error != 0:
                raise Exception(
                    f"Failed to write min position limit for motor {self.id}"
                )

        finally:
            # Restore torque state if it was enabled
            if torque_enabled:
                self.enable_torque()

    def get_position(self):
        """Get current position of the motor."""
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.id, self.ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to get position for motor {self.id}")
        return position

    def set_position(self, position, smoothness=0):
        """
        Set goal position for the motor with optional smoothness.

        Args:
            position (int): Target position.
            smoothness (int): Smoothness level (0 for no smoothing, higher values for more smoothing).
        """
        if self.min_position is None or self.max_position is None:
            self.read_position_limits()

        if position < self.min_position or position > self.max_position:
            raise ValueError(
                f"Position {position} out of range ({self.min_position}-{self.max_position}) "
                f"for motor {self.id}"
            )

        if smoothness <= 0:
            # No smoothing; move directly to the target position
            result, error = self.packet_handler.write4ByteTxRx(
                self.port_handler, self.id, self.ADDR_GOAL_POSITION, position
            )
            if result != COMM_SUCCESS or error != 0:
                raise Exception(f"Failed to set position for motor {self.id}")
        else:
            # Apply smoothing by interpolating positions
            current_position = self.get_position()
            step = max(
                1, int(self.range / (smoothness * 10))
            )  # Calculate step size based on smoothness level
            delay = 0.01 / smoothness  # Calculate delay based on smoothness

            direction = 1 if position > current_position else -1
            for pos in range(current_position, position, direction * step):
                self._write_position(pos)
                time.sleep(delay)

            # Ensure the final position is set accurately
            self._write_position(position)

    def _write_position(self, position):
        """Internal helper to write a position to the motor."""
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.id, self.ADDR_GOAL_POSITION, position
        )
        if result != COMM_SUCCESS or error != 0:
            raise Exception(f"Failed to set position for motor {self.id}")


class XM430W350(DynamixelMotorBase):
    """
    Class for XM430-W350 motor.
    """

    def __init__(self, motor_id, port_handler, packet_handler):
        super().__init__(motor_id, port_handler, packet_handler)
        # Default limits for XM430-W350
        self.DEFAULT_MIN_POSITION = 0
        self.DEFAULT_MAX_POSITION = 4095

    def set_default_limits(self):
        """Set default position limits for XM430-W350."""
        self.set_position_limits(self.DEFAULT_MIN_POSITION, self.DEFAULT_MAX_POSITION)


class XL430W250(DynamixelMotorBase):
    """
    Class for XL430-W250 motor.
    """

    def __init__(self, motor_id, port_handler, packet_handler):
        super().__init__(motor_id, port_handler, packet_handler)
        # Default limits for XL430-W250
        self.DEFAULT_MIN_POSITION = 0
        self.DEFAULT_MAX_POSITION = 4095

    def set_default_limits(self):
        """Set default position limits for XL430-W250."""
        self.set_position_limits(self.DEFAULT_MIN_POSITION, self.DEFAULT_MAX_POSITION)


class DynamixelController:
    """
    Controller class to manage multiple Dynamixel motors.
    """

    def __init__(self, port="COM11", baudrate=57600):
        """
        Initialize the controller.

        Args:
            port (str): Serial port
            baudrate (int): Communication speed
        """
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(DynamixelMotorBase.PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {port}")
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate to {baudrate}")

        self.motors = {}

    def add_motor(self, motor: DynamixelMotorBase):
        """
        Add a motor to the controller.

        Args:
            motor (DynamixelMotorBase): Motor instance
        """
        self.motors[motor.id] = motor
        return motor

    def set_multiple_positions(self, goals):
        """
        Set positions for multiple motors simultaneously with individual smoothness settings.

        Args:
            goals (dict): Dictionary with motor IDs as keys and a sub-dictionary as values.
                        Sub-dictionary format: {"position": int, "smoothness": int}.
        """
        # Separate smoothness=0 cases (direct movement) from smooth movement
        direct_goals = {}
        smooth_goals = {}

        for motor_id, goal in goals.items():
            position = goal["position"]
            smoothness = goal.get("smoothness", 0)

            if smoothness <= 0:
                direct_goals[motor_id] = position
            else:
                smooth_goals[motor_id] = {
                    "position": position,
                    "smoothness": smoothness,
                }

        # Handle direct goals with synchronized write
        if direct_goals:
            group_sync_write = GroupSyncWrite(
                self.port_handler,
                self.packet_handler,
                DynamixelMotorBase.ADDR_GOAL_POSITION,
                4,
            )

            for motor_id, position in direct_goals.items():
                param_goal_position = [
                    DXL_LOBYTE(DXL_LOWORD(position)),
                    DXL_HIBYTE(DXL_LOWORD(position)),
                    DXL_LOBYTE(DXL_HIWORD(position)),
                    DXL_HIBYTE(DXL_HIWORD(position)),
                ]
                result = group_sync_write.addParam(motor_id, param_goal_position)
                if not result:
                    raise Exception(f"Failed to add parameter for motor {motor_id}")

            result = group_sync_write.txPacket()
            if result != COMM_SUCCESS:
                raise Exception("Failed to execute synchronized movement")

            group_sync_write.clearParam()

        # Handle smooth goals
        if smooth_goals:
            current_positions = {
                motor_id: self.motors[motor_id].get_position()
                for motor_id in smooth_goals
            }
            complete = {motor_id: False for motor_id in smooth_goals}

            while not all(complete.values()):
                group_sync_write = GroupSyncWrite(
                    self.port_handler,
                    self.packet_handler,
                    DynamixelMotorBase.ADDR_GOAL_POSITION,
                    4,
                )

                for motor_id, goal in smooth_goals.items():
                    if complete[motor_id]:
                        continue

                    target_position = goal["position"]
                    smoothness = goal["smoothness"]
                    current_position = current_positions[motor_id]

                    step = max(
                        1, abs(target_position - current_position) // (smoothness * 10)
                    )
                    direction = 1 if target_position > current_position else -1

                    next_position = current_position + direction * step
                    if (direction == 1 and next_position >= target_position) or (
                        direction == -1 and next_position <= target_position
                    ):
                        next_position = target_position
                        complete[motor_id] = True

                    current_positions[motor_id] = next_position

                    param_goal_position = [
                        DXL_LOBYTE(DXL_LOWORD(next_position)),
                        DXL_HIBYTE(DXL_LOWORD(next_position)),
                        DXL_LOBYTE(DXL_HIWORD(next_position)),
                        DXL_HIBYTE(DXL_HIWORD(next_position)),
                    ]
                    result = group_sync_write.addParam(motor_id, param_goal_position)
                    if not result:
                        raise Exception(f"Failed to add parameter for motor {motor_id}")

                result = group_sync_write.txPacket()
                if result != COMM_SUCCESS:
                    raise Exception("Failed to execute synchronized movement")

                group_sync_write.clearParam()
                time.sleep(
                    0.01 / min(goal["smoothness"] for goal in smooth_goals.values())
                )

    def close(self):
        """Close the port and clean up."""
        self.port_handler.closePort()


if __name__ == "__main__":
    try:
        # Initialize controller
        port_win = "COM11"  # for Windows
        port_ubuntu = "/dev/ttyUSB0"  # for Linux
        controller = DynamixelController(port=port_ubuntu)

        # Motors
        motor_0 = XM430W350(0, controller.port_handler, controller.packet_handler)
        motor_0.set_position_limits(1024, 3072)
        motor_0.disable_torque()

        # motor_1 = XM430W350(1, controller.port_handler, controller.packet_handler)
        # motor_1.set_position_limits(1024, 3072)
        # motor_1.disable_torque()

        # motor_2 = XL430W250(2, controller.port_handler, controller.packet_handler)
        # motor_2.set_position_limits(2850, 3200)
        # motor_2.disable_torque()

        # Add motors with their specific types
        motor0 = controller.add_motor(motor_0)
        # motor1 = controller.add_motor(motor_1)
        # motor2 = controller.add_motor(motor_2)

        # Enable all torque
        motor_0.enable_torque()
        # motor_1.enable_torque()
        # motor_2.enable_torque()

        # Move motors to their middle position
        motor_0.set_position(motor_0.min_position + int(motor_0.range * 0.5), 5)
        # motor_1.set_position(motor_1.min_position + int(motor_1.range * 0.5), 5)
        # motor_2.set_position(motor_2.min_position + int(motor_2.range * 0.0), 5)
        # goals_middle = {
        #     motor_0.id: {
        #         "position": motor_0.min_position + int(motor_0.range * 0.5),
        #         "smoothness": 2,
        #     },
        #     motor_1.id: {
        #         "position": motor_1.min_position + int(motor_1.range * 0.5),
        #         "smoothness": 2,
        #     },
        #     motor_2.id: {
        #         "position": motor_2.min_position + int(motor_2.range * 0.0),
        #         "smoothness": 2,
        #     },
        # }
        # controller.set_multiple_positions(goals_middle)

        # Wait for movements to complete
        time.sleep(2)

        # Move motors to their target position
        motor_0.set_position(motor_0.min_position + int(motor_0.range * 0.75), 5)
        # motor_1.set_position(motor_1.min_position + int(motor_1.range * 0.75), 5)
        # motor_2.set_position(motor_2.min_position + int(motor_2.range * 0.0), 5)
        # goals_target = {
        #     motor_0.id: {
        #         "position": motor_0.min_position + int(motor_0.range * 0.25),
        #         "smoothness": 3,
        #     },
        #     motor_1.id: {
        #         "position": motor_1.min_position + int(motor_1.range * 0.25),
        #         "smoothness": 3,
        #     },
        #     motor_2.id: {
        #         "position": motor_2.min_position + int(motor_2.range * 0.0),
        #         "smoothness": 3,
        #     },
        # }
        # controller.set_multiple_positions(goals_target)

        # Wait for movements to complete
        time.sleep(1)

        # Move claw open then close
        # motor_2.set_position(motor_2.max_position, 0)
        # time.sleep(1)
        # motor_2.set_position(motor_2.min_position, 0)

        # Finish movement
        time.sleep(5)
        for motor_id, motor in controller.motors.items():
            motor.disable_torque()

    except Exception as e:
        print(f"Error: {e}")
