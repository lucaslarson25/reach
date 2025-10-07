# REACH Project Structure

This document explains the organization of the REACH codebase.

## Directory Layout

```
reach/
├── documentation/          # Project documentation and deliverables
│   ├── headshots/         # Team photos
│   ├── logos/             # Project branding
│   └── *.pdf              # Project documents
│
├── website/               # Project website (HTML/CSS)
│   ├── assets/           # Website assets (images, CSS)
│   └── *.html            # Website pages
│
├── src/                  # Main source code (Python package)
│   ├── simulation/       # MuJoCo environments
│   │   ├── models/       # MuJoCo XML model files
│   │   ├── tasks/        # Task-specific environments
│   │   ├── arm_environment.py
│   │   ├── rewards.py
│   │   └── sensors.py
│   │
│   ├── agents/           # RL agents (PPO, SAC)
│   │   ├── ppo_agent.py
│   │   └── sac_agent.py
│   │
│   ├── vision/           # YOLO integration
│   │   ├── yolo_detector.py
│   │   └── camera.py
│   │
│   ├── control/          # Control policies
│   │   ├── policy.py
│   │   └── controllers.py
│   │
│   └── utils/            # Utilities
│       ├── config.py
│       ├── logger.py
│       └── math_utils.py
│
├── config/               # Configuration files (YAML)
│   └── default.yaml     # Default experiment config
│
├── scripts/              # Executable scripts
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   ├── visualize.py     # Visualization script
│   └── slurm_train.sh   # SLURM job script for Monsoon
│
├── tests/                # Unit tests
│   └── test_*.py        # Test files
│
├── notebooks/            # Jupyter notebooks
│   └── *.ipynb          # Experiment notebooks
│
├── models/               # Saved model checkpoints
│   └── .gitkeep         # (models are gitignored)
│
├── logs/                 # Training logs
│   └── .gitkeep         # (logs are gitignored)
│
├── data/                 # Datasets (if any)
│   └── (gitignored)
│
├── requirements.txt      # Python dependencies
├── setup.py             # Package installation
├── .gitignore           # Git ignore rules
└── README.md            # Main project README

```

## Development Workflow

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install project as package (editable mode)
pip install -e .
```

### 2. Running Experiments Locally
```bash
# Train an agent
python scripts/train.py --config config/default.yaml

# Evaluate a trained model
python scripts/evaluate.py --model models/final_model.zip --n_episodes 100

# Visualize environment or policy
python scripts/visualize.py --model models/final_model.zip
```

### 3. Running on Monsoon HPC
```bash
# Submit training job
sbatch scripts/slurm_train.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/slurm_*.out
```

### 4. Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_environment.py

# Run with coverage
pytest tests/ --cov=src
```

## Code Organization Principles

1. **Separation of Concerns**
   - `simulation/` - Environment logic only
   - `agents/` - RL algorithm implementations
   - `vision/` - Perception/computer vision
   - `control/` - Low-level control
   - `utils/` - Shared utilities

2. **Configuration Management**
   - All experiments configured via YAML files in `config/`
   - No hardcoded hyperparameters in code
   - Easy to version and share configurations

3. **Modularity**
   - Each module is self-contained
   - Clear interfaces between modules
   - Easy to swap implementations (e.g., PPO → SAC)

4. **Reproducibility**
   - Configuration files track all experiment settings
   - Random seeds for reproducibility
   - Model checkpoints saved with configs

## Adding New Components

### New Task Environment
1. Create file in `src/simulation/tasks/`
2. Inherit from `ArmEnvironment`
3. Implement task-specific reset, step, reward
4. Add to `src/simulation/tasks/__init__.py`

### New RL Algorithm
1. Create file in `src/agents/`
2. Implement training interface (train, save, load, predict)
3. Add to `src/agents/__init__.py`
4. Create config section in YAML

### New Utility
1. Add to appropriate file in `src/utils/`
2. Export in `src/utils/__init__.py`
3. Document with docstrings

## Next Steps

After reviewing this skeleton structure, you should:

1. **Implement Core Environment**
   - Create MuJoCo XML model for robotic arm
   - Implement `ArmEnvironment` class
   - Test in notebook or with visualize script

2. **Implement Simple Task**
   - Start with `ReachingTask`
   - Define reward function
   - Test that it runs

3. **Train First Agent**
   - Implement `PPOAgent` wrapper
   - Run training with default config
   - Validate learning is occurring

4. **Iterate and Expand**
   - Add more complex tasks
   - Integrate vision (YOLO)
   - Tune hyperparameters
   - Deploy on Monsoon for longer training

