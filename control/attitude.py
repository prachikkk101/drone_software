"""
Outer-loop attitude controller.

Converts desired attitude (roll, pitch, yaw) in radians to desired
angular rates (p, q, r) in rad/s using three independent PID controllers.

Gains are intentionally conservative for initial simulation tuning.

    Roll  PID: kp=6.0, ki=0.0, kd=0.3  → produces desired roll rate (rad/s)
    Pitch PID: kp=6.0, ki=0.0, kd=0.3  → produces desired pitch rate (rad/s)
    Yaw   PID: kp=4.0, ki=0.0, kd=0.2  → produces desired yaw rate (rad/s)

Output limits: ±5 rad/s (≈286 deg/s) to prevent aggressive manoeuvres.
"""

import math

from control.pid import PID


def _wrap_angle(angle):
    """Wrap angle to [-π, +π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class AttitudeController:
    """
    Outer attitude loop.

    Parameters
    ----------
    dt : float, optional
        Default timestep (seconds).  Can be overridden per call.
    """

    def __init__(self):
        # --- Roll PID ---
        # kp=6.0: moderate proportional gain
        # ki=0.0: no integral (avoids wind-up in first tuning pass)
        # kd=0.3: light damping
        self.roll_pid = PID(
            kp=6.0, ki=0.0, kd=0.3,
            output_min=-5.0, output_max=5.0,
        )

        # --- Pitch PID --- (symmetric with roll for a symmetric quad)
        self.pitch_pid = PID(
            kp=6.0, ki=0.0, kd=0.3,
            output_min=-5.0, output_max=5.0,
        )

        # --- Yaw PID --- (lower gain; yaw inertia is larger)
        self.yaw_pid = PID(
            kp=4.0, ki=0.0, kd=0.2,
            output_min=-5.0, output_max=5.0,
        )

    def update(self, desired_roll, desired_pitch, desired_yaw,
               current_roll, current_pitch, current_yaw, dt):
        """
        Compute desired angular rates from attitude error.

        Parameters
        ----------
        desired_roll, desired_pitch, desired_yaw : float
            Target attitude in radians.
        current_roll, current_pitch, current_yaw : float
            Estimated/current attitude in radians.
        dt : float
            Timestep in seconds.

        Returns
        -------
        (desired_p, desired_q, desired_r) : tuple of float
            Desired body angular rates in rad/s.
        """
        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch
        # Yaw wraps around ±π; roll/pitch typically stay within ±π/2
        # during normal flight so wrapping is not needed.
        yaw_error = _wrap_angle(desired_yaw - current_yaw)

        desired_p = self.roll_pid.update(roll_error, dt)
        desired_q = self.pitch_pid.update(pitch_error, dt)
        desired_r = self.yaw_pid.update(yaw_error, dt)

        return desired_p, desired_q, desired_r

    def reset(self):
        """Reset all PID integrators and derivative state."""
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
