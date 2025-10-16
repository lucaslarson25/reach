import time
from envs.mujoco_arm_env import ReacherEnv

# Create the environment with rendering
env = ReacherEnv(render_mode="human")

# Reset the environment
obs, info = env.reset()
print("Initial observation:", obs)

# Run for a few seconds to observe motion
for step in range(200):
    # Apply random small actions
    action = env.action_space.sample() * 0.3
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        print(f"Target reached at step {step}")
        env.reset()
    
    # Slow down simulation a bit
    time.sleep(0.02)

env.close()
print("Closed environment.")