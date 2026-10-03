#pragma once

class PID
{
public:

    PID(
        double kp,
        double ki,
        double kd,
        double min_output,
        double max_output
    );

    double update(
        double error,
        double dt
    );

    void reset();

private:

    double kp_;
    double ki_;
    double kd_;

    double min_output_;
    double max_output_;

    double integral_;
    double previous_error_;

    bool initialized_;
};