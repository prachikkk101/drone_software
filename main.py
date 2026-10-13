"""
VegaPilot — Python-only attitude-response simulation (Milestone 2).

This is an **attitude-only** simulation. There are no translational or
altitude dynamics. The collective thrust is initialized to mass * gravity
solely to establish a hover-like motor baseline for the mixer; it does NOT
demonstrate physical hovering or flight readiness.

Cascaded control loop
---------------------
    desired attitude
        → AttitudeController  → desired angular rates
        → RateController      → desired body torques
        → QuadMixer           → per-motor thrust targets (N)
        → thrust_to_command conversion → normalized motor-speed commands [0, 1]
        → MotorModel          → actual thrusts & reaction torques
        → body torque reconstruction  → (tx, ty, tz) applied to plant
        → QuadDynamics.step() → updated attitude & angular rates

Motor ordering and geometry (X-configuration, top view looking down)
--------------------------------------------------------------------
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

Body torque reconstruction from actual motor outputs
----------------------------------------------------
    T_total = T0 + T1 + T2 + T3
    tx      = d * (-T0 + T1 + T2 - T3)   (roll torque)
    ty      = d * (-T0 - T1 + T2 + T3)   (pitch torque)
    tz      = +Q0 - Q1 + Q2 - Q3         (yaw reaction torque)

    where Ti = actual thrust of motor i (N), Qi = reaction torque of motor i (N*m).
    Signs match the mixer matrix exactly.
"""

import csv
import math
import os
import sys

import numpy as np
import yaml

from analysis.plots import plot_attitude
from control.attitude import AttitudeController
from control.mixer import QuadMixer
from control.rate import RateController
from sim.dynamics import QuadDynamics
from sim.motors import MotorModel


def load_config(path="config/vehicle.yaml"):
    """Load vehicle configuration from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def thrust_to_command(thrust_n, max_thrust):
    """
    Convert a thrust target (N) to a normalized motor-speed command in [0, 1].

    The physical motor model is:
        omega = command * max_rpm * (2*pi/60)
        thrust = kf * omega^2 = kf * (max_rpm * 2*pi/60)^2 * command^2
               = max_thrust * command^2

    Inverting:
        command = sqrt(clip(thrust / max_thrust, 0.0, 1.0))

    Parameters
    ----------
    thrust_n : array-like or float
        Per-motor thrust targets in Newtons.
    max_thrust : float
        Maximum single-motor thrust in Newtons (kf * max_omega^2).

    Returns
    -------
    np.ndarray
        Normalized motor-speed commands in [0, 1].
    """
    thrust_n = np.asarray(thrust_n)
    ratio = np.clip(thrust_n / max_thrust, 0.0, 1.0)
    return np.sqrt(ratio)


def compute_body_torques(thrusts, reaction_torques, arm_length):
    """
    Reconstruct body torques from actual motor thrusts and reaction torques.

    Uses the projected lever arm d = arm_length / sqrt(2) and identical sign
    conventions as the mixer matrix in control/mixer.py:
        Roll  tx = d * (-T0 + T1 + T2 - T3)
        Pitch ty = d * (-T0 - T1 + T2 + T3)
        Yaw   tz = +Q0 - Q1 + Q2 - Q3

    Parameters
    ----------
    thrusts : np.ndarray, shape (4,)
        Actual thrust from each motor in Newtons.
    reaction_torques : np.ndarray, shape (4,)
        Reaction torque (about body Z) from each motor in N*m.
    arm_length : float
        Radial center-to-motor distance in metres.

    Returns
    -------
    (tx, ty, tz) : tuple of float
        Reconstructed body torques in N*m.
    """
    T0, T1, T2, T3 = thrusts
    Q0, Q1, Q2, Q3 = reaction_torques
    d = arm_length / math.sqrt(2.0)

    tx = d * (-T0 + T1 + T2 - T3)
    ty = d * (-T0 - T1 + T2 + T3)
    tz = +Q0 - Q1 + Q2 - Q3

    return tx, ty, tz


def run_simulation(show_plot=False):
    """
    Run the attitude-response simulation.

    Parameters
    ----------
    show_plot : bool
        If True, display the plot interactively.

    Returns
    -------
    log : list of dict
        Logged data for every timestep.
    """
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = load_config()
    veh = cfg["vehicle"]

    mass = veh["mass"]              # kg
    arm_length = veh["arm_length"]  # m (radial center-to-motor distance)
    gravity = veh["gravity"]        # m/s²
    Ixx = veh["inertia"]["Ixx"]     # kg·m²
    Iyy = veh["inertia"]["Iyy"]
    Izz = veh["inertia"]["Izz"]
    max_rpm = float(veh["motors"]["max_rpm"])
    kf = float(veh["propeller"]["thrust_coefficient"])   # N / (rad/s)²
    km = float(veh["propeller"]["torque_coefficient"])    # N·m / (rad/s)²

    # Derived physical constants
    torque_to_thrust_ratio = km / kf   # dimensionless ratio (km / kf)
    max_omega = max_rpm * (2.0 * math.pi / 60.0)   # rad/s ≈ 523.599 rad/s
    max_thrust_per_motor = kf * max_omega ** 2     # N ≈ 4.112 N
    total_max_thrust = 4.0 * max_thrust_per_motor  # N ≈ 16.449 N

    # ------------------------------------------------------------------
    # 2. Instantiate modules using YAML parameters
    # ------------------------------------------------------------------
    dynamics = QuadDynamics(Ixx=Ixx, Iyy=Iyy, Izz=Izz)
    motor_model = MotorModel(
        max_rpm=max_rpm,
        thrust_coefficient=kf,
        torque_coefficient=km,
    )
    mixer = QuadMixer(
        arm_length=arm_length,
        torque_to_thrust_ratio=torque_to_thrust_ratio,
        max_thrust=max_thrust_per_motor,
    )
    # PID gains are kept strictly unchanged as required for this baseline
    att_ctrl = AttitudeController()
    rate_ctrl = RateController()

    # ------------------------------------------------------------------
    # 3. Simulation parameters
    # ------------------------------------------------------------------
    dt = 0.001        # s
    t_end = 5.0       # s
    steps = int(t_end / dt)

    # Hover-like collective thrust baseline.
    # NOTE: Rotational dynamics only. Collective thrust is used solely
    # to establish a hover-like motor operating point for the mixer.
    collective_thrust = mass * gravity   # N ≈ 5.886 N

    # Desired attitude schedule.
    target_roll_rad = 0.0
    target_pitch_rad = 0.0
    target_yaw_rad = 0.0
    step_time = 1.0                            # s
    step_roll_deg = 10.0                       # deg
    step_roll_rad = math.radians(step_roll_deg)

    # ------------------------------------------------------------------
    # 4. Simulation loop
    # ------------------------------------------------------------------
    log = []
    saturation_event_count = 0

    for i in range(steps):
        t = i * dt

        # Apply step command at t = 1.0 s
        if t >= step_time:
            target_roll_rad = step_roll_rad

        state = dynamics.state

        # Outer loop: attitude error → desired rates (rad/s)
        des_p, des_q, des_r = att_ctrl.update(
            target_roll_rad, target_pitch_rad, target_yaw_rad,
            state.roll, state.pitch, state.yaw,
            dt,
        )

        # Inner loop: rate error → desired body torques (N*m)
        des_tx, des_ty, des_tz = rate_ctrl.update(
            des_p, des_q, des_r,
            state.p, state.q, state.r,
            dt,
        )

        # Mixer: torques → per-motor thrust targets (N)
        motor_thrusts_target = mixer.mix(
            collective_thrust, des_tx, des_ty, des_tz,
        )

        # Convert thrust targets to normalized motor-speed commands [0, 1]
        commands = thrust_to_command(motor_thrusts_target, max_thrust_per_motor)
        commands = np.clip(commands, 0.0, 1.0)

        # Check for saturation (mixer allocation or command clipping)
        step_saturated = bool(
            mixer.last_allocation_saturated
            or np.any(motor_thrusts_target <= 0.0)
            or np.any(motor_thrusts_target >= max_thrust_per_motor)
        )
        if step_saturated:
            saturation_event_count += 1

        # Motor model: commands → actual outputs (RPM, N, N*m)
        rpm, actual_thrusts, actual_reaction_torques = motor_model.update(commands)

        # Reconstruct body torques from actual motor outputs
        actual_tx, actual_ty, actual_tz = compute_body_torques(
            actual_thrusts, actual_reaction_torques, arm_length,
        )

        # Dynamics step using reconstructed physical body torques
        dynamics.step(
            torques=(actual_tx, actual_ty, actual_tz),
            dt=dt,
        )

        # Logging
        log.append({
            "time": t,
            "desired_roll_deg": math.degrees(target_roll_rad),
            "desired_pitch_deg": math.degrees(target_pitch_rad),
            "desired_yaw_deg": math.degrees(target_yaw_rad),
            "actual_roll_deg": math.degrees(state.roll),
            "actual_pitch_deg": math.degrees(state.pitch),
            "actual_yaw_deg": math.degrees(state.yaw),
            "p": state.p,
            "q": state.q,
            "r": state.r,
            "cmd_0": commands[0],
            "cmd_1": commands[1],
            "cmd_2": commands[2],
            "cmd_3": commands[3],
            "actual_tx": actual_tx,
            "actual_ty": actual_ty,
            "actual_tz": actual_tz,
            "saturated": step_saturated,
        })

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(results_dir, "simulation.csv")
    fieldnames = list(log[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)
    print(f"CSV saved to {csv_path}")

    # Plot
    time_s = [row["time"] for row in log]
    plot_attitude(
        time_s=time_s,
        desired_roll_deg=[row["desired_roll_deg"] for row in log],
        desired_pitch_deg=[row["desired_pitch_deg"] for row in log],
        desired_yaw_deg=[row["desired_yaw_deg"] for row in log],
        actual_roll_deg=[row["actual_roll_deg"] for row in log],
        actual_pitch_deg=[row["actual_pitch_deg"] for row in log],
        actual_yaw_deg=[row["actual_yaw_deg"] for row in log],
        save_path=os.path.join(results_dir, "attitude.png"),
        show=show_plot,
    )

    # ------------------------------------------------------------------
    # 6. Response metrics calculation & summary
    # ------------------------------------------------------------------
    post_step_rolls = [r["actual_roll_deg"] for r in log if r["time"] >= step_time]
    peak_roll_deg = max(post_step_rolls)
    overshoot_percent = max(0.0, (peak_roll_deg - step_roll_deg) / step_roll_deg * 100.0)

    final_roll_deg = log[-1]["actual_roll_deg"]
    final_error_deg = abs(final_roll_deg - step_roll_deg)

    peak_p = max(abs(r["p"]) for r in log)
    all_cmds = [r[k] for r in log for k in ["cmd_0", "cmd_1", "cmd_2", "cmd_3"]]
    min_cmd = min(all_cmds)
    max_cmd = max(all_cmds)

    hover_thrust_single = collective_thrust / 4.0
    hover_command = math.sqrt(hover_thrust_single / max_thrust_per_motor)

    print("\n=== Simulation complete ===")
    print(f"Duration: {t_end} s | Timestep: {dt} s | Total steps: {steps}")
    print(f"Max thrust per motor: {max_thrust_per_motor:.3f} N (Total max: {total_max_thrust:.3f} N)")
    print(f"Normalized motor-speed command for hover: {hover_command:.3f} (not a measured throttle %)")
    print(f"Motor-command range during run: [{min_cmd:.3f}, {max_cmd:.3f}]")
    print(f"Control saturation events: {saturation_event_count} / {steps} steps")
    print("--- Attitude Response Metrics ---")
    print(f"Target roll:             {step_roll_deg:.3f}°")
    print(f"Peak roll:               {peak_roll_deg:.3f}°")
    print(f"Transient overshoot:     {overshoot_percent:.2f}% (above {step_roll_deg}°)")
    print(f"Final roll:              {final_roll_deg:.3f}°")
    print(f"Final steady-state error:{final_error_deg:.3f}°")
    print(f"Final pitch:             {log[-1]['actual_pitch_deg']:.6f}°")
    print(f"Final yaw:               {log[-1]['actual_yaw_deg']:.6f}°")
    print(f"Peak roll rate |p|:      {peak_p:.3f} rad/s ({math.degrees(peak_p):.3f}°/s)")
    print()
    print("PHYSICAL MODEL LIMITATIONS:")
    print("- Rotational attitude dynamics ONLY; no translational or altitude dynamics.")
    print("- Collective thrust (mass * gravity) sets motor operating point for the mixer.")
    print("- Linearized Euler rate kinematics (roll_dot = p); small-angle approximation.")
    print("- Noise-free, uncalibrated provisional propeller coefficients.")
    print("DO NOT treat this simulation as demonstrating flight readiness or physical hovering.")

    return log


if __name__ == "__main__":
    run_simulation(show_plot="--show" in sys.argv)
