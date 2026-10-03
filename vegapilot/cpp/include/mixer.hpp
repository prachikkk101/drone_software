#pragma once

#include "types.hpp"

class QuadMixer
{
public:

    QuadMixer(
        double arm_length,
        double yaw_ratio,
        double max_motor_thrust
    );

    MotorCommand mix(
        double total_thrust,
        const TorqueCommand& torque
    );

private:

    double arm_length_;
    double yaw_ratio_;
    double max_motor_thrust_;
};