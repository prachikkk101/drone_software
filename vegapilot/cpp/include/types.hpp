#pragma once

#include <array>

struct AttitudeState
{
    double roll;
    double pitch;
    double yaw;

    double p;
    double q;
    double r;
};

struct AttitudeCommand
{
    double roll;
    double pitch;
    double yaw;
};

struct TorqueCommand
{
    double tx;
    double ty;
    double tz;
};

struct MotorCommand
{
    std::array<double, 4> thrust;
    std::array<double, 4> normalized;
};