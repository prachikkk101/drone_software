#include "controller.hpp"

#include <cmath>

namespace
{
    constexpr double PI = 3.14159265358979323846;
}

AttitudeController::AttitudeController()

    :

    // Outer attitude loop
    roll_angle_(
        4.0,
        0.0,
        0.0,
        -3.0,
        3.0
    ),

    pitch_angle_(
        4.0,
        0.0,
        0.0,
        -3.0,
        3.0
    ),

    yaw_angle_(
        2.0,
        0.0,
        0.0,
        -2.0,
        2.0
    ),

    // Inner angular-rate loop
    roll_rate_(
        0.05,
        0.02,
        0.001,
        -0.25,
        0.25
    ),

    pitch_rate_(
        0.05,
        0.02,
        0.001,
        -0.25,
        0.25
    ),

    yaw_rate_(
        0.05,
        0.01,
        0.001,
        -0.15,
        0.15
    )
{
}

double AttitudeController::wrapAngle(
    double angle
)
{
    while (angle > PI)
        angle -= 2.0 * PI;

    while (angle < -PI)
        angle += 2.0 * PI;

    return angle;
}

TorqueCommand AttitudeController::update(
    const AttitudeState& state,
    const AttitudeCommand& command,
    double dt
)
{
    /*
        OUTER LOOP

        desired attitude
              ↓
        desired angular rate
    */

    double roll_error =
        wrapAngle(command.roll - state.roll);

    double pitch_error =
        wrapAngle(command.pitch - state.pitch);

    double yaw_error =
        wrapAngle(command.yaw - state.yaw);

    double desired_p =
        roll_angle_.update(
            roll_error,
            dt
        );

    double desired_q =
        pitch_angle_.update(
            pitch_error,
            dt
        );

    double desired_r =
        yaw_angle_.update(
            yaw_error,
            dt
        );

    /*
        INNER LOOP

        desired rate
              ↓
        torque
    */

    double tx =
        roll_rate_.update(
            desired_p - state.p,
            dt
        );

    double ty =
        pitch_rate_.update(
            desired_q - state.q,
            dt
        );

    double tz =
        yaw_rate_.update(
            desired_r - state.r,
            dt
        );

    return {
        tx,
        ty,
        tz
    };
}

void AttitudeController::reset()
{
    roll_angle_.reset();
    pitch_angle_.reset();
    yaw_angle_.reset();

    roll_rate_.reset();
    pitch_rate_.reset();
    yaw_rate_.reset();
}