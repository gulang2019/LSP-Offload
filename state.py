from enum import Enum

class State(Enum):
    RUNNING = 0
    RUNNING_PROFILE = 1
    PROFILE_COMPUTE = 2
    PROFILE_COMM = 3
    PROFILE = 4