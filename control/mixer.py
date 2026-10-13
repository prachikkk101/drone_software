"""
Control allocation mixer for an X-configuration quadcopter.

Motor ordering and geometry (Top view, looking down):
----------------------------------------------------
                    Front (+X)
        Motor 2 (Front-Left)     Motor 3 (Front-Right)
                  \\                 /
                   \\               /
                    \\    [BODY]   /
                     \\           /
                      \\         /
        Motor 1 (Rear-Left)      Motor 0 (Rear-Right)
                    Rear (-X)
Left (-Y)                                Right (+Y)

Coordinate frame:
    +X : Forward (Front)
    +Y : Right (Starboard)
    +Z : Down (FRD / Aerospace standard)

Projected lever arm:
    arm_length (L) is the radial center-to-motor distance.
    For an X-configuration with motors at 45-degree diagonals:
        d = L / sqrt(2) ≈ 0.7071 * L

Motor positions and rotor spin directions:
    Motor 0: Rear-Right  (x = -d, y = +d), CCW rotor -> reaction torque +Q0 on body (+Z)
    Motor 1: Rear-Left   (x = -d, y = -d), CW rotor  -> reaction torque -Q1 on body (-Z)
    Motor 2: Front-Left  (x = +d, y = -d), CCW rotor -> reaction torque +Q2 on body (+Z)
    Motor 3: Front-Right (x = +d, y = +d), CW rotor  -> reaction torque -Q3 on body (-Z)

Allocation matrix mapping motor thrusts [T0, T1, T2, T3]^T to [T_total, tx, ty, tz]^T:
    T_total = T0 + T1 + T2 + T3
    tx      = d * (-T0 + T1 + T2 - T3)   (roll torque)
    ty      = d * (-T0 - T1 + T2 + T3)   (pitch torque)
    tz      = r * (+T0 - T1 + T2 - T3)   (yaw reaction torque, r = km / kf)
"""

import numpy as np


class QuadMixer:
    """
    X-configuration mixer mapping requested total thrust and body torques
    to individual motor thrust targets.
    """

    def __init__(
        self,
        arm_length,
        torque_to_thrust_ratio,
        max_thrust=None,
    ):
        """
        Parameters
        ----------
        arm_length : float
            Radial distance from center of mass to motor in metres.
        torque_to_thrust_ratio : float
            Ratio of motor reaction torque to thrust (km / kf).
        max_thrust : float, optional
            Maximum achievable thrust per motor in Newtons.
        """
        self.arm_length = arm_length
        self.ratio = torque_to_thrust_ratio
        self.max_thrust = max_thrust
        self.lever_arm = arm_length / np.sqrt(2.0)

        # Allocation matrix: maps [T0, T1, T2, T3]^T to [T, tx, ty, tz]^T
        d = self.lever_arm
        r = self.ratio
        self.matrix = np.array([
            [1.0,  1.0,  1.0,  1.0],
            [-d,    d,    d,   -d ],
            [-d,   -d,    d,    d ],
            [ r,   -r,    r,   -r ],
        ])

        # Saturation state from the most recent allocation call
        self.last_allocation_saturated = False

    def mix(
        self,
        total_thrust,
        roll_torque,
        pitch_torque,
        yaw_torque,
    ):
        """
        Solve for per-motor thrusts to achieve desired total thrust and body torques.

        Detects saturation when requested values demand negative motor thrust
        or exceed max_thrust (if configured), setting `last_allocation_saturated`.

        Parameters
        ----------
        total_thrust : float
            Desired collective thrust in Newtons.
        roll_torque : float
            Desired roll torque (tx) in N*m.
        pitch_torque : float
            Desired pitch torque (ty) in N*m.
        yaw_torque : float
            Desired yaw torque (tz) in N*m.

        Returns
        -------
        motor_thrust : np.ndarray, shape (4,)
            Allocated motor thrust targets in Newtons, clamped to achievable limits.
        """
        desired = np.array([
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        ], dtype=float)

        unconstrained_thrust = np.linalg.solve(
            self.matrix,
            desired,
        )

        is_negative = np.any(unconstrained_thrust < 0.0)
        is_excessive = (self.max_thrust is not None) and np.any(unconstrained_thrust > self.max_thrust)
        self.last_allocation_saturated = bool(is_negative or is_excessive)

        max_limit = self.max_thrust if self.max_thrust is not None else np.inf
        return np.clip(
            unconstrained_thrust,
            0.0,
            max_limit,
        )