# Getting Started with REACH - Beginner's Guide

This guide explains the key concepts and workflow for team members new to reinforcement learning, robotics simulation, or the REACH project.

---

## 📚 **Core Concepts Explained**

### What is Reinforcement Learning (RL)?

Think of RL like teaching a dog tricks:
- **Agent** (the dog/robot): Tries different actions
- **Environment** (the world): Responds to those actions  
- **Reward** (treats): Tells the agent if it did well or poorly
- **Learning**: Over time, the agent learns which actions get the most rewards

In REACH:
- **Agent**: The AI controlling the robotic arm
- **Environment**: The simulated world with the arm, objects, and physics
- **Reward**: Points for getting closer to the target, penalties for wasted energy
- **Goal**: Learn to reach for objects, grasp them, and assist with daily tasks

### What is MuJoCo?

MuJoCo is a **physics engine** - software that simulates realistic movement and forces.

Think of it like a video game physics engine:
- Simulates gravity, collisions, joint limits
- Calculates how the arm moves when you apply forces
- Provides "sensors" to measure positions, velocities, contact forces
- Runs much faster than real-time (important for learning)

**Why we use it**: Training a robot in the real world takes forever and can break hardware. Simulation is fast, safe, and free.

### What is Gymnasium?

Gymnasium is a **standard interface** for RL environments.

It defines a simple contract:
```python
# The environment provides these 3 main functions:

observation, info = env.reset()          # Start a new episode
observation, reward, done, ... = env.step(action)  # Take an action
env.render()                              # Visualize (optional)
```

**Why we use it**: All RL algorithms (PPO, SAC, etc.) expect this interface. If we follow it, we can use any RL library.

### What is PPO (Proximal Policy Optimization)?

PPO is the **AI algorithm** that learns how to control the robot.

Simple explanation:
1. The robot tries random actions at first
2. Good actions (that get high rewards) are reinforced
3. Bad actions are discouraged
4. Over millions of tries, it figures out the best strategy (called a "policy")

**Why we use PPO**:
- Stable and reliable
- Works well for continuous control (smooth arm movements)
- Industry standard for robotics

### What is a Neural Network Policy?

The **policy** is the "brain" that decides what action to take.

It's a neural network (like ChatGPT, but much simpler):
- **Input**: Current observations (joint angles, positions, etc.)
- **Output**: Actions to take (joint torques or position targets)
- **Training**: Adjusted to maximize total reward

Think of it as a function:
```
action = policy(observation)
```

The neural network learns this function through trial and error.

---

## 🔄 **The Complete Workflow**

### Phase 1: Design the Environment

**What**: Define how the robot simulation works

**Steps**:
1. Create MuJoCo XML model (`simple_arm.xml`)
   - Define robot structure (joints, links, dimensions)
   - Set physics properties (mass, friction, damping)
   - Add objects and sensors

2. Implement environment class (`arm_environment.py`)
   - Define what the robot "sees" (observations)
   - Define what the robot can "do" (actions)
   - Define what it's rewarded for (reward function)

**Key File**: `src/simulation/arm_environment.py`

---

### Phase 2: Define the Task

**What**: Specify what we want the robot to learn

**Steps**:
1. Pick a task (start simple):
   - **Reaching**: Move end-effector to a target position
   - **Grasping**: Pick up an object
   - **Daily task**: Brush teeth, reach for cup, etc.

2. Design reward function:
   - Positive rewards: Getting closer to goal, completing task
   - Negative rewards: Wasting energy, moving too fast, collisions

**Key File**: `src/simulation/tasks/reaching.py`

**Example Reward**:
```python
# Closer to target = higher reward
distance_to_target = np.linalg.norm(ee_pos - target_pos)
reward = -distance_to_target  # Negative distance (maximize to minimize distance)

# Bonus for success
if distance_to_target < 0.05:  # Within 5cm
    reward += 10.0  # Big success bonus!
```

---

### Phase 3: Train the Agent

**What**: Run the RL algorithm to learn the task

**Steps**:
1. Configure hyperparameters (`config/default.yaml`)
   - Learning rate: How fast the AI learns
   - Batch size: How many experiences to learn from at once
   - Network size: How complex the neural network is

2. Run training script:
   ```bash
   python scripts/train.py --config config/reaching_example.yaml
   ```

3. Monitor progress:
   - TensorBoard: Watch reward increasing over time
   - Checkpoints: Save model every N steps
   - Evaluation: Test current policy periodically

**Key File**: `scripts/train.py`

**What happens during training**:
```
Episode 1: Random actions → Total reward: -250 (terrible)
Episode 100: Getting better → Total reward: -100
Episode 1000: Learning! → Total reward: -50
Episode 5000: Pretty good → Total reward: 5
Episode 10000: Success! → Total reward: 20
```

---

### Phase 4: Evaluate and Visualize

**What**: Test the trained agent and see if it works

**Steps**:
1. Load trained model:
   ```bash
   python scripts/evaluate.py --model models/final_model.zip
   ```

2. Visualize policy:
   ```bash
   python scripts/visualize.py --model models/final_model.zip
   ```

3. Analyze results:
   - Success rate: What % of episodes succeed?
   - Average reward: How well does it perform?
   - Trajectory smoothness: Are movements natural?

**Key Files**: 
- `scripts/evaluate.py`
- `scripts/visualize.py`
- `notebooks/01_environment_testing.ipynb`

---

## 📂 **Key Files and Their Purpose**

### Configuration

| File | Purpose | When to Edit |
|------|---------|--------------|
| `config/default.yaml` | All hyperparameters and settings | Before each experiment |
| `config/reaching_example.yaml` | Example task-specific config | Use as template |

### Simulation

| File | Purpose | When to Edit |
|------|---------|--------------|
| `src/simulation/arm_environment.py` | Main environment class | Once, at the start |
| `src/simulation/tasks/reaching.py` | Task-specific logic | For each new task |
| `src/simulation/rewards.py` | Reward function definitions | When tuning rewards |
| `src/simulation/models/simple_arm.xml` | MuJoCo robot model | When designing arm |

### Training

| File | Purpose | When to Edit |
|------|---------|--------------|
| `src/agents/ppo_agent.py` | PPO algorithm wrapper | Rarely (use as-is) |
| `scripts/train.py` | Training script | Rarely (use as-is) |
| `scripts/slurm_train.sh` | Monsoon HPC job | When deploying to cluster |

### Analysis

| File | Purpose | When to Edit |
|------|---------|--------------|
| `scripts/evaluate.py` | Test trained models | Rarely (use as-is) |
| `scripts/visualize.py` | Visualize behavior | Rarely (use as-is) |
| `notebooks/01_environment_testing.ipynb` | Interactive testing | Often (for debugging) |

---

## 🚀 **Step-by-Step: Your First Experiment**

### Step 1: Setup Environment (5 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (uncomment lines in requirements.txt first!)
pip install -r requirements.txt

# Install project as package
pip install -e .
```

### Step 2: Test MuJoCo Installation (2 minutes)

```python
# In Python shell or notebook
import mujoco
print("MuJoCo version:", mujoco.__version__)
# Should print something like: MuJoCo version: 2.3.7
```

### Step 3: Create Simple MuJoCo Model (30 minutes)

Edit `src/simulation/models/simple_arm.xml`:
- Start with the 2-DOF example provided
- Gradually add more joints to match your real arm
- Test loading: `python -c "import mujoco; mujoco.MjModel.from_xml_path('src/simulation/models/simple_arm.xml')"`

### Step 4: Implement Environment (2-4 hours)

Fill in `src/simulation/arm_environment.py`:

```python
def __init__(self, model_path, task_config, render_mode=None):
    # 1. Load MuJoCo model
    self.model = mujoco.MjModel.from_xml_path(model_path)
    self.data = mujoco.MjData(self.model)
    
    # 2. Define observation space (what the robot "sees")
    # Example: [joint_pos1, joint_pos2, ..., target_x, target_y, target_z]
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
    )
    
    # 3. Define action space (what the robot can "do")
    # Example: joint torques for 2 joints
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )
```

### Step 5: Test Environment (30 minutes)

Use the testing notebook:
```bash
jupyter notebook notebooks/01_environment_testing.ipynb
```

Run through the cells to verify:
- Environment loads without errors
- Observations have correct shape
- Actions are applied correctly
- Rendering works

### Step 6: Implement Simple Reward (1 hour)

In `src/simulation/tasks/reaching.py`:

```python
def step(self, action):
    # Apply action and step physics
    self.data.ctrl[:] = action
    mujoco.mj_step(self.model, self.data)
    
    # Get end-effector position
    ee_pos = self.data.site('end_effector').xpos
    
    # Calculate reward (simple distance-based)
    distance = np.linalg.norm(ee_pos - self.target_pos)
    reward = -distance  # Minimize distance
    
    # Check if task succeeded
    success = distance < 0.05  # Within 5cm
    if success:
        reward += 10.0  # Success bonus!
    
    return observation, reward, success, False, {}
```

### Step 7: Train First Model (1-2 hours)

```bash
# Edit config/reaching_example.yaml if needed
# Then run training
python scripts/train.py --config config/reaching_example.yaml
```

Watch TensorBoard:
```bash
tensorboard --logdir logs/
# Open http://localhost:6006 in browser
```

Look for:
- **Reward increasing**: Agent is learning!
- **Success rate increasing**: Getting better at the task
- **Loss decreasing**: Policy is converging

### Step 8: Evaluate Results (15 minutes)

```bash
# Test the trained model
python scripts/evaluate.py --model models/reaching/final_model.zip --n_episodes 20

# Visualize behavior
python scripts/visualize.py --model models/reaching/final_model.zip
```

---

## 🎯 **Common Issues and Solutions**

### Issue: "ModuleNotFoundError: No module named 'mujoco'"

**Solution**: Install MuJoCo
```bash
pip install mujoco
```

### Issue: Rewards aren't increasing

**Possible causes**:
1. **Reward function too sparse**: Add reward shaping (intermediate rewards)
2. **Learning rate too high/low**: Try 3e-4, 1e-3, or 1e-4
3. **Environment too hard**: Start with easier task first
4. **Bug in code**: Check observations and actions are correct

### Issue: Training is too slow

**Solutions**:
1. Use multiple parallel environments (`n_envs: 8` in config)
2. Decrease `n_steps` (but may hurt performance)
3. Use GPU if available
4. Deploy to Monsoon HPC cluster

### Issue: Agent learns weird behavior

**Causes**:
1. **Reward function is wrong**: Agent optimizes for what you measure, not what you want
2. **Action space too large**: Constrain actions
3. **Observations incomplete**: Agent can't see what it needs

**Solution**: Carefully design reward to encourage desired behavior

---

## 📖 **Learning Resources**

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Best RL introduction
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - Our RL library
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean RL implementations

### MuJoCo
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Forum](https://github.com/google-deepmind/mujoco/discussions)
- [Tutorial](https://pab47.github.io/mujoco.html) - Basics of MuJoCo

### Python/ML
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Gymnasium Docs](https://gymnasium.farama.org/)

---

## ❓ **FAQ**

**Q: Do I need to understand all the math?**  
A: No! Start with the high-level concepts. You can learn the math as you go.

**Q: How long does training take?**  
A: For simple tasks: 10-30 minutes. Complex tasks: hours to days. Use Monsoon for long runs.

**Q: Can I test on the real robot?**  
A: Eventually! Start in simulation, then transfer to real hardware (sim-to-real transfer).

**Q: What if my environment doesn't work?**  
A: Use the testing notebook! Test each component (reset, step, render) separately.

**Q: Should I use PPO or SAC?**  
A: Start with PPO. It's more stable and easier to tune.

**Q: How do I know if my agent is learning?**  
A: Watch TensorBoard. Reward should increase over time. If it plateaus or decreases, something's wrong.

---

## 🤝 **Getting Help**

1. **Check documentation** first (this file, PROJECT_STRUCTURE.md, CONTRIBUTING.md)
2. **Use the testing notebook** to debug issues
3. **Ask in team chat** - someone may have solved the problem
4. **Check GitHub issues** in relevant libraries
5. **Ask faculty mentor** for high-level guidance
6. **Ask sponsors** for domain-specific questions

---

## 🎉 **You're Ready!**

Start with small steps:
1. Get MuJoCo working
2. Load the simple arm model
3. Implement basic environment
4. Train on reaching task
5. Gradually add complexity

**Remember**: Everyone starts as a beginner. The skeleton is designed to guide you. Follow the TODO comments, use the examples, and don't hesitate to ask questions!

Good luck! 🚀

