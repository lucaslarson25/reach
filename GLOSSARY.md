# REACH Project Glossary

Quick reference for common terms used in reinforcement learning, robotics, and the REACH project.

---

## Reinforcement Learning Terms

### Agent
The AI that learns to perform tasks. In our case, it's the neural network policy that controls the robotic arm.

### Environment  
The simulated world that the agent interacts with. For us, this is the MuJoCo physics simulation with the robotic arm.

### Observation / State
What the agent "sees" at each timestep. Examples:
- Joint angles: `[0.5, 1.2, -0.3, ...]`
- End-effector position: `[0.3, 0.2, 1.1]`
- Target location: `[0.5, 0.1, 1.3]`

### Action
What the agent can "do" to affect the environment. Examples:
- Joint torques: `[-0.5, 0.8, 0.2, ...]`
- Joint position targets: `[1.0, 0.5, -0.2, ...]`

### Reward
Numerical feedback telling the agent how well it's doing.
- Positive reward: good behavior (moving toward target)
- Negative reward / penalty: bad behavior (wasting energy)
- Total reward over episode measures overall performance

### Episode
One complete sequence from start to finish:
1. Reset environment (random start position, random target)
2. Agent takes actions
3. Episode ends when task succeeds OR timeout occurs

### Policy (π)
The strategy the agent uses to choose actions. Implemented as a neural network:
```
action = policy(observation)
```

### Value Function (V)
Estimates how good a state is (expected total future reward from this state).

### Discount Factor (γ, gamma)
How much to value future rewards vs immediate rewards:
- γ = 0: only care about immediate reward
- γ = 1: future rewards equally important
- γ = 0.99: standard value (balances short and long term)

### Exploration vs Exploitation
- **Exploration**: Try random actions to discover new strategies
- **Exploitation**: Use what you've learned to maximize reward
- Need balance: too much exploration = never get good, too much exploitation = miss better strategies

### On-Policy vs Off-Policy
- **On-policy** (PPO): Learn from data collected by current policy
- **Off-policy** (SAC): Can learn from old data (replay buffer)
- PPO is more stable, SAC is more sample-efficient

---

## Deep Learning Terms

### Neural Network
A mathematical function (like the brain) that learns patterns from data. Composed of layers of neurons.

### Policy Network
Neural network that implements the policy:
- Input: observation (sensor data)
- Output: action (what to do)

### Value Network  
Neural network that estimates how good a state is (used in PPO).

### Gradient
Direction to adjust network weights to improve performance. Think of it as "which way to climb the hill."

### Learning Rate
How big of steps to take when updating the network. 
- Too high: unstable, misses optimum
- Too low: learns very slowly

### Batch Size
How many examples to learn from at once. Larger batches = more stable but slower.

### Epoch
One pass through the entire dataset. In RL, how many times to reuse collected experience.

### Overfitting
When model memorizes training data instead of learning general patterns. Performs well on training but poorly on new situations.

### Hyperparameters
Settings that control the learning process (learning rate, batch size, etc.). Not learned from data - you must set them.

---

## MuJoCo / Physics Terms

### MuJoCo
Multi-Joint dynamics with Contact. A physics engine for robotics simulation.

### XML Model
File format MuJoCo uses to define robots, environments, and physics properties.

### Joint
Connection between two rigid bodies (links) in the robot. Types:
- **Hinge**: rotates around one axis (like elbow)
- **Slide**: translates along one axis  
- **Ball**: rotates in all directions (like shoulder)
- **Free**: moves and rotates freely

### Link / Body
Rigid segment of the robot (upper arm, forearm, etc.)

### Actuator
Motor or controller that applies force/torque to joints.

### Geom (Geometry)
Shape used for visualization and collision detection (box, sphere, cylinder, mesh).

### Contact
When two geometries touch. MuJoCo computes contact forces automatically.

### End-Effector
The "hand" or tool at the end of the robotic arm that interacts with objects.

### Degrees of Freedom (DOF)
Number of independent ways the robot can move. A 7-DOF arm has 7 joints it can control.

### Forward Kinematics
Given joint angles → compute end-effector position.

### Inverse Kinematics  
Given desired end-effector position → compute required joint angles.

### Workspace
Region of 3D space the end-effector can reach.

---

## Stable-Baselines3 (SB3) Terms

### Stable-Baselines3
Python library providing pre-implemented RL algorithms (PPO, SAC, etc.). Saves us from coding algorithms from scratch.

### Vectorized Environment (VecEnv)
Runs multiple environments in parallel for faster data collection.

### Callback
Function called during training to:
- Save checkpoints
- Evaluate performance
- Record videos
- Log metrics

### Model
In SB3, the complete agent including:
- Policy network
- Value network (if applicable)
- Training state (optimizer, buffers, etc.)

---

## PPO-Specific Terms

### Proximal Policy Optimization (PPO)
Our primary RL algorithm. Key features:
- On-policy (learns from current policy)
- Clip prevents large policy changes
- Stable and reliable for robotics

### Clip Range
Limits how much the policy can change in one update. Prevents destructive updates. Standard value: 0.2

### GAE (Generalized Advantage Estimation)
Method to estimate how good an action was compared to average. Balances bias and variance.

### GAE Lambda (λ)
Parameter controlling GAE computation. Standard value: 0.95

### Entropy Bonus
Encourages exploration by rewarding random actions. Higher entropy = more random policy.

---

## Computer Vision Terms

### YOLO (You Only Look Once)
Fast object detection algorithm. Given image → outputs bounding boxes and classes:
```
Input: RGB image
Output: [
  {class: "cup", bbox: [100, 200, 50, 60], confidence: 0.95},
  {class: "spoon", bbox: [300, 150, 40, 80], confidence: 0.87}
]
```

### Bounding Box
Rectangle around detected object: `[x, y, width, height]`

### Confidence Score
How certain the detector is. 0.0 = not sure, 1.0 = very confident.

### RGB Image
Color image with 3 channels: Red, Green, Blue.

### Depth Map
Image where each pixel value represents distance from camera.

### RGBD
Image with both color (RGB) and depth (D) information.

### Camera Intrinsics
Camera properties: focal length, principal point. Needed to convert 2D pixels to 3D positions.

---

## Project-Specific Terms

### REACH
Reinforcement learning for assistive robotic arm control (our project).

### Sim-to-Real Transfer
Training in simulation, then deploying to real robot. Challenging because:
- Sim and real have different physics
- Sensors have noise in real world
- Real world has unexpected variations

### Monsoon
NAU's High-Performance Computing (HPC) cluster. Used for long training runs.

### SLURM
Job scheduler for Monsoon. Manages who uses which computers when.

### TensorBoard
Tool for visualizing training progress (reward curves, loss, etc.).

### Weights & Biases (W&B)
Cloud-based experiment tracking. Like TensorBoard but online.

---

## File Extensions

### `.xml`
MuJoCo model file. Defines robot structure and physics.

### `.yaml` / `.yml`  
Configuration file. Human-readable format for settings.

### `.py`
Python source code file.

### `.ipynb`
Jupyter notebook. Interactive Python with code, text, and plots.

### `.zip` / `.pkl` / `.pth`
Saved model file (trained agent).

---

## Common Abbreviations

### RL
Reinforcement Learning

### DRL  
Deep Reinforcement Learning (RL with neural networks)

### MLP
Multi-Layer Perceptron (standard neural network architecture)

### CNN
Convolutional Neural Network (for processing images)

### LSTM
Long Short-Term Memory (type of recurrent neural network)

### HPC
High-Performance Computing (Monsoon cluster)

### EE
End-Effector

### DOF
Degrees of Freedom

### FPS
Frames Per Second (how fast simulation/rendering runs)

### GPU
Graphics Processing Unit (accelerates neural network training)

### CPU
Central Processing Unit

### API
Application Programming Interface (how different code pieces communicate)

---

## Typical Values Reference

### Learning Rates
- Very low: `1e-5` (0.00001)
- Low: `1e-4` (0.0001)
- Standard: `3e-4` (0.0003)
- High: `1e-3` (0.001)
- Very high: `1e-2` (0.01)

### Discount Factors
- Short-term: `0.9` to `0.95`
- Standard: `0.99`
- Long-term: `0.995` to `0.999`

### Batch Sizes
- Small: 32-64
- Medium: 128-256
- Large: 512-1024

### Network Sizes
- Small: `[64, 64]`
- Medium: `[128, 128]` or `[256, 256]`
- Large: `[512, 512]` or `[256, 256, 256]`

### Episode Lengths
- Simple tasks: 100-500 steps
- Medium tasks: 500-2000 steps
- Complex tasks: 2000+ steps

---

## Quick Conversions

### Timesteps to Real Time
- MuJoCo timestep typically `0.002` seconds
- 500 steps = 1 second of simulation
- 1000 steps = 2 seconds

### Training Duration Estimates
- Simple task: 500K - 1M steps (~1-2 hours)
- Medium task: 2M - 5M steps (~4-10 hours)
- Complex task: 10M+ steps (days)

(Times assume 4-8 parallel environments on decent CPU)

---

## Questions to Ask When Stuck

### Reward not increasing?
- Is reward function correct? Does it encourage what you want?
- Is learning rate appropriate? Try different values.
- Is task too hard? Simplify first, then increase difficulty.
- Are observations sufficient? Can the agent "see" what it needs?

### Agent learns weird behavior?
- Check reward function - agent optimizes exactly what you measure
- Ensure reward encourages desired behavior at each step
- Consider adding constraints or safety penalties

### Training too slow?
- Use more parallel environments (`n_envs`)
- Use GPU if available
- Simplify environment if possible
- Deploy to Monsoon for long runs

### Simulation doesn't match expectations?
- Check MuJoCo XML model properties (mass, friction, limits)
- Verify sensor readings make sense
- Test environment in isolation before training

---

This glossary should help you understand the terminology used throughout the codebase and documentation!

