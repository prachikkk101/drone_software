import numpy as np


class QuadState:

    def __init__(self):

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.p = 0.0
        self.q = 0.0
        self.r = 0.0

    def vector(self):

        return np.array([
            self.roll,
            self.pitch,
            self.yaw,
            self.p,
            self.q,
            self.r,
        ])

    def load(self, x):

        (
            self.roll,
            self.pitch,
            self.yaw,
            self.p,
            self.q,
            self.r,
        ) = x


class QuadDynamics:

    def __init__(
        self,
        Ixx=0.005,
        Iyy=0.005,
        Izz=0.009,
    ):

        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

        self.state = QuadState()

    def derivatives(
        self,
        x,
        torques,
    ):

        roll, pitch, yaw, p, q, r = x

        tx, ty, tz = torques

        p_dot = (
            tx
            + (self.Iyy - self.Izz)
            * q * r
        ) / self.Ixx

        q_dot = (
            ty
            + (self.Izz - self.Ixx)
            * p * r
        ) / self.Iyy

        r_dot = (
            tz
            + (self.Ixx - self.Iyy)
            * p * q
        ) / self.Izz

        roll_dot = p
        pitch_dot = q
        yaw_dot = r

        return np.array([
            roll_dot,
            pitch_dot,
            yaw_dot,
            p_dot,
            q_dot,
            r_dot,
        ])

    def step(
        self,
        torques,
        dt,
    ):

        x = self.state.vector()

        # RK4 integration

        k1 = self.derivatives(
            x,
            torques,
        )

        k2 = self.derivatives(
            x + 0.5 * dt * k1,
            torques,
        )

        k3 = self.derivatives(
            x + 0.5 * dt * k2,
            torques,
        )

        k4 = self.derivatives(
            x + dt * k3,
            torques,
        )

        new_x = x + (
            dt / 6.0
        ) * (
            k1
            + 2*k2
            + 2*k3
            + k4
        )

        self.state.load(new_x)

        return self.state