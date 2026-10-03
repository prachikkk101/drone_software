#include <iostream>
#include <sstream>

#include "controller.hpp"
#include "mixer.hpp"

int main()
{
    constexpr double mass = 0.6;
    constexpr double gravity = 9.81;

    constexpr double arm_length = 0.13;

    constexpr double kf = 2.0e-7;
    constexpr double km = 5.0e-9;
    constexpr double max_rpm = 5000.0;

    const double max_motor_thrust =
        kf * max_rpm * max_rpm;

    const double yaw_ratio = km / kf;

    AttitudeController controller;

    QuadMixer mixer(
        arm_length,
        yaw_ratio,
        max_motor_thrust
    );

    const double hover_thrust =
        mass * gravity;

    double roll;
    double pitch;
    double yaw;

    double p;
    double q;
    double r;

    double command_roll;
    double command_pitch;
    double command_yaw;

    double dt;

    while (
        std::cin
        >> roll
        >> pitch
        >> yaw
        >> p
        >> q
        >> r
        >> command_roll
        >> command_pitch
        >> command_yaw
        >> dt
    )
    {
        AttitudeState state{
            roll,
            pitch,
            yaw,
            p,
            q,
            r
        };

        AttitudeCommand command{
            command_roll,
            command_pitch,
            command_yaw
        };

        TorqueCommand torque =
            controller.update(
                state,
                command,
                dt
            );

        MotorCommand motors =
            mixer.mix(
                hover_thrust,
                torque
            );

        std::cout
            << motors.normalized[0] << " "
            << motors.normalized[1] << " "
            << motors.normalized[2] << " "
            << motors.normalized[3]
            << std::endl;
    }

    return 0;
}