import mss
import cv2
import numpy as np
import time
from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO

env = Z1ReachEnv(render_mode="human")
model = PPO.load("ppo_z1_parallel", env=env)

output_video = "ppo_z1_run.mp4"
frame_rate = 60
frame_width, frame_height = 640, 480

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))

with mss.mss() as sct:
    monitor = sct.monitors[1]  # primary screen

    for ep in range(3):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or (time.time() - start_time > 30)

            # Capture screen
            img = np.array(sct.grab(monitor))
            frame = cv2.resize(img[:, :, :3], (frame_width, frame_height))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            time.sleep(1 / frame_rate)

video_writer.release()
env.close()
