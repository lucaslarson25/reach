# Contributing to REACH

This document provides guidelines for team members contributing to the REACH project.

## Git Workflow

### Branching Strategy

We use a feature-branch workflow:

1. **Main branches:**
   - `main` - Production-ready code, stable releases
   - `develop` - Integration branch for features

2. **Feature branches:**
   - Create from `develop`
   - Name format: `feature/description` (e.g., `feature/mujoco-environment`)
   - Name format: `fix/description` for bug fixes
   - Name format: `experiment/description` for experimental work

### Workflow Steps

```bash
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# 2. Make changes and commit
git add .
git commit -m "Descriptive commit message"

# 3. Push to remote
git push origin feature/your-feature-name

# 4. Create Pull Request on GitHub
# - Request review from at least one team member
# - Address any feedback
# - Merge when approved
```

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
Good:
- "Implement ReachingTask environment with MuJoCo"
- "Fix reward calculation in grasping task"
- "Add PPO training script for Monsoon HPC"

Bad:
- "Update"
- "Fix bug"
- "Changes"
```

## Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) style guidelines:

- **Indentation:** 4 spaces (no tabs)
- **Line length:** Max 100 characters
- **Naming:**
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Docstrings:** Use Google-style docstrings

### Example:

```python
def calculate_reward(distance, smoothness, energy):
    """
    Calculate reward for reaching task.
    
    Args:
        distance: Distance to target (float)
        smoothness: Smoothness penalty (float)
        energy: Energy consumption (float)
    
    Returns:
        reward: Combined reward value
    """
    reward = -distance - 0.1 * smoothness - 0.01 * energy
    return reward
```

### Code Formatting

We use `black` for automatic code formatting:

```bash
# Format entire codebase
black src/ scripts/ tests/

# Format specific file
black src/simulation/arm_environment.py
```

### Linting

Check code quality with `flake8`:

```bash
flake8 src/ scripts/ tests/
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

Aim for >80% code coverage for core modules:
- Simulation environments
- Reward functions
- Agent implementations
- Utility functions

## Documentation

### Docstrings

Every module, class, and function should have a docstring:

```python
class ArmEnvironment(gym.Env):
    """
    MuJoCo environment for robotic arm simulation.
    
    This environment simulates a wearable robotic arm and provides
    a Gymnasium interface for reinforcement learning.
    
    Attributes:
        model: MuJoCo model object
        data: MuJoCo data object
        action_space: Gymnasium action space
        observation_space: Gymnasium observation space
    """
```

### Code Comments

- Comment complex logic
- Explain **why**, not just **what**
- Use TODO comments for future work

```python
# Good comment:
# Use running normalization to stabilize training
# (observation values can vary widely across tasks)
obs = self.normalize_observation(obs)

# Bad comment:
# Normalize observation
obs = self.normalize_observation(obs)
```

## Pull Request Process

1. **Create PR** on GitHub from your feature branch to `develop`
2. **Description:** Clearly describe what changes were made and why
3. **Tests:** Ensure all tests pass
4. **Review:** Request review from at least one team member
5. **Address feedback:** Make requested changes
6. **Merge:** Once approved, merge using "Squash and merge"

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Tested manually

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No unnecessary files committed
```

## Project-Specific Guidelines

### Configuration Management

- **Never hardcode** hyperparameters in code
- Add all parameters to YAML config files
- Document new config parameters in comments

### Experiment Tracking

- Use descriptive experiment names
- Save all experiment configs
- Log results to TensorBoard
- Document findings in notebooks or docs

### MuJoCo Models

- Place XML files in `src/simulation/models/`
- Document model changes in model README
- Version models (arm_v1.xml, arm_v2.xml, etc.)

### Monsoon HPC

- Test locally before submitting to cluster
- Use appropriate resource requests
- Monitor job progress
- Clean up old job outputs

## Communication

### Team Meetings

- Weekly mentor meetings: Thursdays 4:30-5:30 PM
- Sponsor meetings: Bi-weekly Tuesdays 2:00-3:30 PM
- Capstone lectures: Fridays 12:45-3:15 PM

### Asking for Help

- Check existing documentation first
- Post questions in team chat
- Tag specific team members for their expertise
- Document solutions for future reference

## Code Review Guidelines

When reviewing PRs:

✅ **Do:**
- Be constructive and respectful
- Ask questions to understand intent
- Suggest improvements
- Test the changes if possible

❌ **Don't:**
- Nitpick trivial style issues (use linters)
- Block on personal preferences
- Approve without actually reviewing

## Version Control Best Practices

- **Commit frequently** with atomic changes
- **Pull before push** to avoid conflicts
- **Don't commit** large files, generated files, or secrets
- **Use .gitignore** to exclude unnecessary files

## Getting Help

- **Documentation:** Check README.md and PROJECT_STRUCTURE.md
- **Team members:** See team.html for roles and expertise
- **Faculty mentor:** For high-level guidance
- **Sponsors:** For domain-specific questions

## Questions?

If you're unsure about any guidelines, ask in team meetings or chat!

