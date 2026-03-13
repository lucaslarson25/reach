import numpy as np
import mujoco.viewer
from scenes.ainex_soccer.env import AINexReachEnv

env = AINexReachEnv(render_mode=None)

print("Actuators:", env.model.nu)
print("Action space:", env.action_space.shape)

obs, _ = env.reset()

t = 0.0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    while viewer.is_running():

        # action = 0.15 * np.sin(np.arange(env.model.nu) * 0.5 + t)

        # env.step(action)

        viewer.sync()

        t += 0.03