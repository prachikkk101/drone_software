import numpy as np


class MotorModel:

    def __init__(
        self,
        max_rpm=5000.0,
        thrust_coefficient=1.5e-5,
        torque_coefficient=2.5e-7,
    ):
        self.max_rpm = max_rpm
        self.kf = thrust_coefficient
        self.km = torque_coefficient

    def command_to_rpm(self, command):
        """
        Convert normalized motor command [0,1]
        into motor RPM.
        """

        command = np.clip(command, 0.0, 1.0)

        return command * self.max_rpm

    def rpm_to_thrust(self, rpm):
        """
        Propeller thrust model.
        Converts RPM to angular speed omega (rad/s), then computes
        thrust = kf * omega^2 (Newtons), where kf is in N/(rad/s)^2.
        """
        omega = rpm * (2.0 * np.pi / 60.0)
        return self.kf * omega**2

    def rpm_to_torque(self, rpm):
        """
        Reaction torque model.
        Converts RPM to angular speed omega (rad/s), then computes
        reaction_torque = km * omega^2 (N*m), where km is in N*m/(rad/s)^2.
        """
        omega = rpm * (2.0 * np.pi / 60.0)
        return self.km * omega**2

    def update(self, commands):

        commands = np.asarray(commands)

        rpm = self.command_to_rpm(commands)

        thrust = self.rpm_to_thrust(rpm)

        torque = self.rpm_to_torque(rpm)

        return rpm, thrust, torque