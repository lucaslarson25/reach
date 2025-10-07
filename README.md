# REACH Capstone Project

## Project Overview

The REACH Capstone Project is a comprehensive web application developed by a team of Computer Science students at Northern Arizona University. This project represents the culmination of their academic journey, showcasing their skills in web development, project management, and collaborative software engineering.

## Team Members

### Development Team

- **Taylor Davis** (tjd352@nau.edu)
  - **Role:** Team Lead / Coder / Architect
  - **Responsibilities:** Meeting leadership, external communications, strategic direction, and technical contributions

- **Victor Rodriguez** (vr527@nau.edu)
  - **Role:** Coder / Recorder / Architect
  - **Responsibilities:** Task management, documentation, core coding, and architectural oversight
  - **Background:** U.S. Marine Corps veteran with extensive leadership experience

- **Clayton Ramsey** (car723@nau.edu)
  - **Role:** Coder / Architect
  - **Responsibilities:** Development support, collaboration, architectural input, and quality assurance

- **Lucas Larson** (lwl33@nau.edu)
  - **Role:** Coder / Version Control Manager / Architect
  - **Responsibilities:** GitHub operations, process enforcement, coding, and team mentorship

## Project Sponsors

- **Dr. Zach Lerner, Ph.D.**
  - Associate Professor, Mechanical Engineering, NAU
  - Website: [https://biomech.nau.edu](https://biomech.nau.edu)

- **Prof. Carlo R. da Cunha, Ph.D.**
  - Assistant Professor, Electrical Engineering, NAU
  - Website: [https://ac.nau.edu/~cc3682](https://ac.nau.edu/~cc3682)

## Technology Stack

- **Frontend:** HTML5, CSS3, Bootstrap 4.3.1
- **Styling:** Custom CSS with dark mode support
- **Responsive Design:** Mobile-first approach
- **Version Control:** Git/GitHub
- **Documentation:** Markdown, Google Docs, Microsoft Office Suite

## Project Structure

```
reach/
├── README.md                 # Project documentation
├── PROJECT_STRUCTURE.md      # Detailed structure documentation
│
├── website/                  # Project website
│   ├── index.html           # Homepage
│   ├── team.html            # Team information page
│   ├── project.html         # Project details page
│   ├── documents.html       # Project documents page
│   └── assets/              # Website assets (CSS, images, logos)
│
├── documentation/            # Project documentation and deliverables
│   ├── headshots/           # Team headshots
│   ├── logos/               # Project logos and branding
│   └── *.pdf                # Project documents
│
├── src/                      # Main source code (Python package)
│   ├── simulation/          # MuJoCo environments and physics
│   ├── agents/              # RL agents (PPO, SAC)
│   ├── vision/              # YOLO object detection
│   ├── control/             # Control policies and controllers
│   └── utils/               # Shared utilities
│
├── config/                   # Configuration files (YAML)
├── scripts/                  # Training and evaluation scripts
├── tests/                    # Unit and integration tests
├── notebooks/                # Jupyter notebooks for experiments
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
└── .gitignore                # Git ignore rules
```

For detailed information about the codebase structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Features

### Website Features
- **Responsive Design:** Optimized for desktop, tablet, and mobile devices
- **Dark Mode:** Comprehensive dark theme implementation for Bootstrap 4
- **Team Information:** Detailed team member profiles with roles and responsibilities
- **Project Documentation:** Centralized access to project documents and resources
- **Modern UI/UX:** Clean, professional design with smooth animations and transitions

### Technical Features
- **Bootstrap Integration:** Custom dark mode overrides for Bootstrap 4.3.1
- **Cross-browser Compatibility:** Tested across modern web browsers
- **Accessibility:** Semantic HTML and proper contrast ratios
- **Performance:** Optimized CSS and minimal JavaScript dependencies

## Getting Started

### Prerequisites
- **Python 3.9+** for simulation and RL development
- **MuJoCo** physics engine (version 2.3+)
- **Modern web browser** for viewing the project website
- **NAU Monsoon HPC access** for large-scale training (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lucaslarson25/reach.git
   cd reach
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install project as package (development mode):**
   ```bash
   pip install -e .
   ```

### Quick Start

**View the project website:**
```bash
cd website
open index.html  # or use a local web server
```

**Test the simulation environment:**
```bash
python scripts/visualize.py --env config/default.yaml
```

**Train an RL agent:**
```bash
python scripts/train.py --config config/default.yaml
```

**Evaluate a trained model:**
```bash
python scripts/evaluate.py --model models/final_model.zip --n_episodes 100
```

### Development on Monsoon HPC

For training on NAU's Monsoon cluster:

```bash
# Load required modules
module load python/3.10
module load cuda/11.8

# Submit training job
sbatch scripts/slurm_train.sh
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed workflow information.

## Team Collaboration

### Meeting Schedule
- **Weekly Mentor Meetings:** Thursdays, 4:30–5:30 PM
- **Bi-weekly Sponsor Meetings:** Every other Tuesday, 2:00–3:30 PM
- **Weekly Capstone Lectures:** Fridays, 12:45–3:15 PM
- **Impromptu Meetings:** Scheduled with 24-hour notice for urgent issues

### Communication Tools
- **Task Tracker:** Shared project management system
- **GitHub Issues:** Bug tracking and feature requests
- **Version Control:** Git repository with branching strategy
- **Documentation:** Google Docs, Microsoft Word/PowerPoint, Draw.io, Lucidchart

### Decision Making
- **Consensus:** Preferred method for team decisions
- **Majority Vote:** ¾ majority when consensus cannot be reached
- **Escalation:** Faculty mentor resolution for persistent disagreements

## Project Standards

### Documentation Standards
- Google Docs for collaborative documents
- Microsoft Word/PowerPoint for formal presentations
- Draw.io and Lucidchart for diagrams and flowcharts
- Markdown for technical documentation

### Coding Standards
- GitHub branching conventions
- Clean, commented code
- Responsive design principles
- Cross-browser compatibility
- Accessibility guidelines

### Communication Standards
- Professional, respectful communication
- Regular status updates
- Comprehensive meeting minutes
- 24-hour distribution of meeting notes
- Clear action item tracking

## Contributing

This is a capstone project for academic purposes. For questions or contributions, please contact the development team through their respective email addresses listed above.

## License

This project is developed as part of the Computer Science Capstone course at Northern Arizona University. All rights reserved.

## Contact Information

For project inquiries or questions, please contact:
- **Team Lead:** Taylor Davis (tjd352@nau.edu)
- **Faculty Mentor:** [Contact information to be added]
- **Course Coordinator:** [Contact information to be added]

---

**REACH Capstone Project**  
Northern Arizona University  
Computer Science Department  
2024