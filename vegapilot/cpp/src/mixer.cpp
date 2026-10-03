#include "mixer.hpp"

#include <algorithm>
#include <cmath>

QuadMixer::QuadMixer(
    double arm_length,
    double yaw_ratio,
    double max_motor_thrust
)
    : arm_length_(arm_length),
      yaw_ratio_(yaw_ratio),
      max_motor_thrust_(max_motor_thrust)
{
}

MotorCommand QuadMixer::mix(
    double total_thrust,
    const TorqueCommand& torque
)
{
    MotorCommand output{};

    const double L =
        arm_length_ / std::sqrt(2.0);

    /*
       X configuration:

              M1      M2
                \    /
                 \  /
                 /  \
                /    \
              M4      M3

       The exact motor numbering and
       spin directions must remain consistent
       throughout the simulator.
    */

    double T1 =
        total_thrust / 4.0
        - torque.tx / (4.0 * L)
        - torque.ty / (4.0 * L)
        + torque.tz / (4.0 * yaw_ratio_);

    double T2 =
        total_thrust / 4.0
        + torque.tx / (4.0 * L)
        - torque.ty / (4.0 * L)
        - torque.tz / (4.0 * yaw_ratio_);

    double T3 =
        total_thrust / 4.0
        + torque.tx / (4.0 * L)
        + torque.ty / (4.0 * L)
        + torque.tz / (4.0 * yaw_ratio_);

    double T4 =
        total_thrust / 4.0
        - torque.tx / (4.0 * L)
        + torque.ty / (4.0 * L)
        - torque.tz / (4.0 * yaw_ratio_);

    output.thrust = {
        T1,
        T2,
        T3,
        T4
    };

    for (int i = 0; i < 4; ++i)
    {
        output.thrust[i] = std::clamp(
            output.thrust[i],
            0.0,
            max_motor_thrust_
        );

        output.normalized[i] =
            output.thrust[i] /
            max_motor_thrust_;
    }

    return output;
}