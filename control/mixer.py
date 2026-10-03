import numpy as np


class QuadMixer:

    def __init__(
        self,
        arm_length,
        torque_to_thrust_ratio,
    ):
        self.arm_length = arm_length
        self.ratio = torque_to_thrust_ratio

    def mix(
        self,
        total_thrust,
        roll_torque,
        pitch_torque,
        yaw_torque,
    ):

        L = self.arm_length

        # X configuration.
        # This matrix describes how motor thrust
        # contributes to total thrust and torque.

        matrix = np.array([
            [1, 1, 1, 1],

            [-L, L, L, -L],

            [-L, -L, L, L],

            [self.ratio,
             -self.ratio,
             self.ratio,
             -self.ratio],
        ])

        desired = np.array([
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        ])

        motor_thrust = np.linalg.solve(
            matrix,
            desired,
        )

        return np.maximum(
            motor_thrust,
            0.0,
        )