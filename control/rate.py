"""
Inner-loop angular-rate controller.

Converts desired angular rates (p, q, r) in rad/s to desired body torques
(tx, ty, tz) in N·m using three independent PID controllers.

Gains and output limits
-----------------------
    Roll-rate  PID: kp=0.015, ki=0.002, kd=0.0005   output: ±0.05 N·m
    Pitch-rate PID: kp=0.015, ki=0.002, kd=0.0005   output: ±0.05 N·m
    Yaw-rate   PID: kp=0.020, ki=0.003, kd=0.0003   output: ±0.05 N·m

Rationale
---------
- kp ≈ Ixx * desired_bandwidth  →  0.005 * 3 ≈ 0.015
  This gives ~3 rad/s² per rad/s of rate error, a moderate inner-loop bandwidth.
- ki provides steady-state error correction; kept small to avoid integral wind-up.
- kd adds damping; kept very small because the inner loop already differentiates
  the rate error (which is effectively the second derivative of attitude).
- Output torque limits of ±0.05 N·m are physically realistic for 130 mm micro
  quads (arm_length * max_motor_thrust ≈ 0.13 * 0.375 ≈ 0.049 N·m per motor pair).
"""

from control.pid import PID


class RateController:
    """
    Inner angular-rate loop.

    Produces body-frame torques from rate errors.
    """

    def __init__(self):
        # --- Roll-rate PID (about body x-axis) ---
        self.roll_rate_pid = PID(
            kp=0.015, ki=0.002, kd=0.0005,
            output_min=-0.05, output_max=0.05,
        )

        # --- Pitch-rate PID (about body y-axis, symmetric with roll) ---
        self.pitch_rate_pid = PID(
            kp=0.015, ki=0.002, kd=0.0005,
            output_min=-0.05, output_max=0.05,
        )

        # --- Yaw-rate PID (about body z-axis) ---
        # Slightly higher kp because Izz > Ixx, and yaw motor authority
        # comes from reaction torque (smaller moment arm than thrust arm).
        self.yaw_rate_pid = PID(
            kp=0.020, ki=0.003, kd=0.0003,
            output_min=-0.05, output_max=0.05,
        )

    def update(self, desired_p, desired_q, desired_r,
               current_p, current_q, current_r, dt):
        """
        Compute desired body torques from rate error.

        Parameters
        ----------
        desired_p, desired_q, desired_r : float
            Desired angular rates in rad/s.
        current_p, current_q, current_r : float
            Current body angular rates in rad/s.
        dt : float
            Timestep in seconds.

        Returns
        -------
        (tx, ty, tz) : tuple of float
            Desired body torques in N·m.
        """
        p_error = desired_p - current_p
        q_error = desired_q - current_q
        r_error = desired_r - current_r

        tx = self.roll_rate_pid.update(p_error, dt)
        ty = self.pitch_rate_pid.update(q_error, dt)
        tz = self.yaw_rate_pid.update(r_error, dt)

        return tx, ty, tz

    def reset(self):
        """Reset all PID integrators and derivative state."""
        self.roll_rate_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()
