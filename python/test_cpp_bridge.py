from cpp_bridge import CppController


class State:
    roll = 0.0
    pitch = 0.0
    yaw = 0.0

    p = 0.0
    q = 0.0
    r = 0.0


controller = CppController(
    "../cpp/build/Release/vegapilot_rt.exe"
)

state = State()

motors = controller.update(
    state,
    [
        0.1745,
        0.0,
        0.0,
    ],
    0.001,
)

print("Motor commands:")
print(motors)

controller.close()