"""
Automated verification tests for Milestone 2 attitude simulation.

Tests cover:
    1. Motor unit consistency and inverse conversion (Gate 1).
    2. Control allocation, X-configuration lever arms, and saturation (Gate 2).
    3. Transient overshoot, steady-state error, cross-axis isolation, and limits (Gate 3).

Run with:
    python -m pytest tests/test_python_simulation.py -q
"""

import math

import numpy as np
import pytest

from control.mixer import QuadMixer
from main import compute_body_torques, load_config, run_simulation, thrust_to_command
from sim.motors import MotorModel


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def sim_log():
    """Run the 5-second simulation once for all end-to-end assertions."""
    return run_simulation(show_plot=False)


@pytest.fixture(scope="module")
def vehicle_params():
    """Load configuration parameters for vehicle checks."""
    cfg = load_config()
    veh = cfg["vehicle"]
    mass = float(veh["mass"])
    gravity = float(veh["gravity"])
    arm_length = float(veh["arm_length"])
    max_rpm = float(veh["motors"]["max_rpm"])
    kf = float(veh["propeller"]["thrust_coefficient"])
    km = float(veh["propeller"]["torque_coefficient"])
    max_omega = max_rpm * (2.0 * math.pi / 60.0)
    max_thrust = kf * max_omega ** 2
    return {
        "mass": mass,
        "gravity": gravity,
        "arm_length": arm_length,
        "max_rpm": max_rpm,
        "kf": kf,
        "km": km,
        "max_omega": max_omega,
        "max_thrust": max_thrust,
        "total_max_thrust": 4.0 * max_thrust,
        "hover_command": math.sqrt((mass * gravity / 4.0) / max_thrust),
    }


# ==============================================================================
# Gate 1: Motor Units & Mathematical Inverse Tests
# ==============================================================================

class TestMotorUnitsAndHoverConsistency:
    """Verifies motor physical units, thrust calculations, and hover baseline."""

    def test_max_thrust_per_motor(self, vehicle_params):
        """Max thrust per motor should be approx 4.112 N with adopted rad/s units."""
        # kf = 1.5e-5 N/(rad/s)^2, max_omega = 5000 * 2*pi/60 ≈ 523.599 rad/s
        # Expected: 1.5e-5 * (523.599)^2 ≈ 4.1123 N
        assert np.isclose(vehicle_params["max_thrust"], 4.112, atol=0.005), (
            f"Expected max thrust per motor ≈ 4.112 N, got {vehicle_params['max_thrust']:.4f} N"
        )

    def test_total_max_thrust(self, vehicle_params):
        """Total maximum thrust (4 motors) should be approx 16.449 N."""
        assert np.isclose(vehicle_params["total_max_thrust"], 16.449, atol=0.010), (
            f"Expected total max thrust ≈ 16.449 N, got {vehicle_params['total_max_thrust']:.4f} N"
        )

    def test_hover_motor_speed_command(self, vehicle_params):
        """Normalized motor-speed command for hover should be approx 0.598."""
        # W = 0.6 * 9.81 = 5.886 N, per motor = 1.4715 N
        # cmd = sqrt(1.4715 / 4.1123) ≈ 0.5982
        assert np.isclose(vehicle_params["hover_command"], 0.598, atol=0.005), (
            f"Expected hover command ≈ 0.598, got {vehicle_params['hover_command']:.4f}"
        )

    def test_thrust_to_command_exact_inverse(self, vehicle_params):
        """thrust_to_command must be the exact inverse of MotorModel across [0, max_thrust]."""
        model = MotorModel(
            max_rpm=vehicle_params["max_rpm"],
            thrust_coefficient=vehicle_params["kf"],
            torque_coefficient=vehicle_params["km"],
        )
        max_t = vehicle_params["max_thrust"]
        test_commands = np.linspace(0.0, 1.0, 21)

        _, thrusts, _ = model.update(test_commands)
        reconstructed_commands = thrust_to_command(thrusts, max_t)

        np.testing.assert_allclose(
            reconstructed_commands,
            test_commands,
            atol=1e-12,
            err_msg="thrust_to_command is not the exact mathematical inverse of MotorModel",
        )

    def test_thrust_to_command_clipping(self, vehicle_params):
        """Negative thrust targets must clamp to 0; excessive thrust targets must clamp to 1."""
        max_t = vehicle_params["max_thrust"]
        assert thrust_to_command(-5.0, max_t) == 0.0
        assert thrust_to_command(max_t * 2.0, max_t) == 1.0


# ==============================================================================
# Gate 2: Control Allocation & Geometry Tests
# ==============================================================================

class TestControlAllocationAndGeometry:
    """Verifies X-configuration lever arms, forward torque reconstruction, and saturation."""

    def test_unsaturated_allocation_agreement(self, vehicle_params):
        """
        Unsaturated requests must achieve requested total thrust and all 3 torques
        within floating-point tolerance (atol=1e-10).
        """
        arm_length = vehicle_params["arm_length"]
        ratio = vehicle_params["km"] / vehicle_params["kf"]
        max_t = vehicle_params["max_thrust"]

        mixer = QuadMixer(
            arm_length=arm_length,
            torque_to_thrust_ratio=ratio,
            max_thrust=max_t,
        )

        # Realistic unsaturated request
        req_thrust = 5.886   # N (hover thrust)
        req_tx = 0.015       # N*m roll torque
        req_ty = -0.010      # N*m pitch torque
        req_tz = 0.005       # N*m yaw torque

        motor_thrusts = mixer.mix(req_thrust, req_tx, req_ty, req_tz)
        assert not mixer.last_allocation_saturated

        # Forward reconstruction
        achieved_total_thrust = float(np.sum(motor_thrusts))
        reaction_torques = motor_thrusts * ratio
        achieved_tx, achieved_ty, achieved_tz = compute_body_torques(
            motor_thrusts, reaction_torques, arm_length
        )

        assert math.isclose(achieved_total_thrust, req_thrust, abs_tol=1e-10)
        assert math.isclose(achieved_tx, req_tx, abs_tol=1e-10)
        assert math.isclose(achieved_ty, req_ty, abs_tol=1e-10)
        assert math.isclose(achieved_tz, req_tz, abs_tol=1e-10)

    def test_pure_roll_mixer_request_cross_axis_isolation(self, vehicle_params):
        """
        An unsaturated pure-roll mixer request must achieve requested roll torque
        while producing approximately zero pitch and yaw torque (atol=1e-12).
        """
        arm_length = vehicle_params["arm_length"]
        ratio = vehicle_params["km"] / vehicle_params["kf"]
        max_t = vehicle_params["max_thrust"]

        mixer = QuadMixer(
            arm_length=arm_length,
            torque_to_thrust_ratio=ratio,
            max_thrust=max_t,
        )

        req_thrust = 5.886
        req_tx = 0.020  # pure roll torque
        req_ty = 0.0
        req_tz = 0.0

        motor_thrusts = mixer.mix(req_thrust, req_tx, req_ty, req_tz)
        assert not mixer.last_allocation_saturated

        reaction_torques = motor_thrusts * ratio
        achieved_tx, achieved_ty, achieved_tz = compute_body_torques(
            motor_thrusts, reaction_torques, arm_length
        )

        assert math.isclose(achieved_tx, req_tx, abs_tol=1e-10)
        assert math.isclose(achieved_ty, 0.0, abs_tol=1e-12)
        assert math.isclose(achieved_tz, 0.0, abs_tol=1e-12)

    def test_explicit_saturation_detection(self, vehicle_params):
        """Mixer must detect saturation when requested thrust/torques exceed limits."""
        mixer = QuadMixer(
            arm_length=vehicle_params["arm_length"],
            torque_to_thrust_ratio=vehicle_params["km"] / vehicle_params["kf"],
            max_thrust=vehicle_params["max_thrust"],
        )
        # Requesting 10.0 N*m of roll torque on a micro quad is impossible
        mixer.mix(total_thrust=1.0, roll_torque=10.0, pitch_torque=0.0, yaw_torque=0.0)
        assert mixer.last_allocation_saturated is True


# ==============================================================================
# Gate 3: Response Metrics, Cross-Axis Isolation & Simulation State Tests
# ==============================================================================

class TestEndToEndAttitudeSimulation:
    """Verifies end-to-end roll response, cross-axis isolation, and metric definitions."""

    def test_simulation_completes_and_finite(self, sim_log):
        """State values, angular rates, and applied torques must remain finite throughout."""
        assert len(sim_log) == 5000
        for r in sim_log:
            assert math.isfinite(r["actual_roll_deg"])
            assert math.isfinite(r["actual_pitch_deg"])
            assert math.isfinite(r["actual_yaw_deg"])
            assert math.isfinite(r["p"])
            assert math.isfinite(r["q"])
            assert math.isfinite(r["r"])
            assert math.isfinite(r["actual_tx"])
            assert math.isfinite(r["actual_ty"])
            assert math.isfinite(r["actual_tz"])

    def test_motor_commands_bounded(self, sim_log):
        """All motor commands must stay strictly within [0, 1]."""
        for r in sim_log:
            for k in ["cmd_0", "cmd_1", "cmd_2", "cmd_3"]:
                assert 0.0 <= r[k] <= 1.0, f"Command {k}={r[k]} out of [0, 1] at t={r['time']}"

    def test_roll_moves_in_correct_direction(self, sim_log):
        """After step command at t=1.0s, roll must move positively toward 10.0 deg."""
        row_pre = min(sim_log, key=lambda r: abs(r["time"] - 0.5))
        row_post = min(sim_log, key=lambda r: abs(r["time"] - 1.5))
        assert row_pre["actual_roll_deg"] < 0.001
        assert row_post["actual_roll_deg"] > 4.0

    def test_cross_axis_isolation_in_noise_free_model(self, sim_log):
        """
        Pitch and yaw must remain near zero (< 1e-6 deg) throughout the pure roll maneuver
        in this deterministic, ideal model.
        """
        max_pitch = max(abs(r["actual_pitch_deg"]) for r in sim_log)
        max_yaw = max(abs(r["actual_yaw_deg"]) for r in sim_log)
        assert max_pitch < 1e-6, f"Pitch deviated during roll maneuver: max |pitch| = {max_pitch}°"
        assert max_yaw < 1e-6, f"Yaw deviated during roll maneuver: max |yaw| = {max_yaw}°"

    def test_transient_overshoot_and_steady_state_error_definitions(self, sim_log):
        """
        Transient overshoot must be calculated from peak roll after step command (t >= 1.0s)
        and reported separately from final steady-state error.
        """
        post_step = [r["actual_roll_deg"] for r in sim_log if r["time"] >= 1.0]
        peak_roll = max(post_step)
        overshoot_pct = max(0.0, (peak_roll - 10.0) / 10.0 * 100.0)

        final_roll = sim_log[-1]["actual_roll_deg"]
        final_ss_error = abs(final_roll - 10.0)

        # Baseline check: with untuned gains, overshoot is ~18.15% and final error is ~0.022 deg
        assert np.isclose(overshoot_pct, 18.15, atol=0.20), (
            f"Expected baseline overshoot ≈ 18.15%, got {overshoot_pct:.2f}%"
        )
        assert np.isclose(final_ss_error, 0.022, atol=0.005), (
            f"Expected baseline steady-state error ≈ 0.022°, got {final_ss_error:.4f}°"
        )
        # Ensure final error is distinctly smaller than dynamic overshoot
        assert overshoot_pct > 10.0
        assert final_ss_error < 0.1

    def test_headless_execution(self):
        """Simulation must run without opening a GUI plot window."""
        log = run_simulation(show_plot=False)
        assert len(log) > 0
