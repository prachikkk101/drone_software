#pragma once

#include "pid.hpp"
#include "types.hpp"

class AttitudeController
{
public:

    AttitudeController();

    TorqueCommand update(
        const AttitudeState& state,
        const AttitudeCommand& command,
        double dt
    );

    void reset();

private:

    PID roll_angle_;
    PID pitch_angle_;
    PID yaw_angle_;

    PID roll_rate_;
    PID pitch_rate_;
    PID yaw_rate_;

    static double wrapAngle(double angle);
};