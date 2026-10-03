class PID:

    def __init__(
        self,
        kp,
        ki,
        kd,
        output_min=None,
        output_max=None,
    ):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error, dt):

        # Integral
        self.integral += error * dt

        # Derivative
        derivative = (
            error - self.previous_error
        ) / dt

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = error

        if self.output_min is not None:
            output = max(
                output,
                self.output_min,
            )

        if self.output_max is not None:
            output = min(
                output,
                self.output_max,
            )

        return output