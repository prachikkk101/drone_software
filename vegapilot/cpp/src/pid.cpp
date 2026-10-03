#include "pid.hpp"

#include <algorithm>

PID::PID(
    double kp,
    double ki,
    double kd,
    double min_output,
    double max_output
)
    : kp_(kp),
      ki_(ki),
      kd_(kd),
      min_output_(min_output),
      max_output_(max_output),
      integral_(0.0),
      previous_error_(0.0),
      initialized_(false)
{
}

double PID::update(
    double error,
    double dt
)
{
    if (dt <= 0.0)
        return 0.0;

    if (!initialized_)
    {
        previous_error_ = error;
        initialized_ = true;
    }

    double derivative =
        (error - previous_error_) / dt;

    double new_integral =
        integral_ + error * dt;

    double output =
        kp_ * error
        + ki_ * new_integral
        + kd_ * derivative;

    // Anti-windup:
    // only accept the new integral if
    // the controller isn't being pushed
    // further into saturation.

    if (output >= min_output_ &&
        output <= max_output_)
    {
        integral_ = new_integral;
    }

    output =
        kp_ * error
        + ki_ * integral_
        + kd_ * derivative;

    output = std::clamp(
        output,
        min_output_,
        max_output_
    );

    previous_error_ = error;

    return output;
}

void PID::reset()
{
    integral_ = 0.0;
    previous_error_ = 0.0;
    initialized_ = false;
}