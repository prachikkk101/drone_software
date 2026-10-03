import subprocess


class CppController:
    def __init__(self, executable):
        self.process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def update(
        self,
        state,
        command,
        dt,
    ):
        values = [
            state.roll,
            state.pitch,
            state.yaw,
            state.p,
            state.q,
            state.r,
            command[0],
            command[1],
            command[2],
            dt,
        ]

        line = " ".join(
            str(value)
            for value in values
        )

        self.process.stdin.write(
            line + "\n"
        )

        self.process.stdin.flush()

        response = (
            self.process.stdout.readline()
        )

        if not response:
            raise RuntimeError(
                "C++ controller stopped."
            )

        motors = [
            float(x)
            for x in response.split()
        ]

        if len(motors) != 4:
            raise RuntimeError(
                "Invalid motor output."
            )

        return motors

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()

        self.process.terminate()
        self.process.wait()