# LifePlus: Metabolic Cellular Automaton Simulation

LifePlus is a modular, stochastic simulation engine designed to model metabolic life using cellular automata. It features resource management, waste toxicity, agent evolution through mutation, and a real-time dashboard for observation and analysis.

## Features

- **Grid-based Environment**: Tracks resources, waste, and agent occupancy using NumPy for high performance.
- **Evolving Agents**: Agents harvest resources, generate waste, move, and reproduce with mutated genomes.
- **Stochastic Decision Making**: Movement and actions are determined by weighted probabilities (Softmax) rather than rigid logic.
- **Interactive Dashboard**: A Streamlit-based UI for real-time visualization and KPI tracking.
- **Statistical Testing**: A specialized test harness to verify stability in stochastic systems.

## Project Structure

```text
lifePlus/
├── src/
│   ├── agent.py       # Agent logic and genome mutation
│   ├── grid.py        # Grid state management (NumPy)
│   ├── simulation.py  # Orchestration of the tick cycle
│   ├── streamlit.py   # Streamlit UI Dashboard
│   ├── main.py        # CLI demonstration entry point
│   └── tests.py       # Unit and statistical tests
├── README.md          # Project documentation
└── LICENSE            # License information
```

## Prerequisites

- **Python 3.10+** (Developed and tested with Python 3.14)
- **pip** (Python package installer)

## Installation

1.  **Clone the repository** (or download the source):
    ```powershell
    git clone <repository-url>
    cd lifePlus
    ```

2.  **Install dependencies**:
    ```powershell
    pip install numpy streamlit plotly pandas
    ```

## Running the Application

### 1. The Interactive Dashboard (Recommended)
Launch the Streamlit dashboard to observe the simulation in real-time with visualizations:
```powershell
streamlit run src/streamlit.py
```

### 2. CLI Demonstration
Run a quick 5-tick simulation in the terminal:
```powershell
python src/main.py
```
*(Note: If `python` is not in your path, use `py src/main.py` on Windows)*

## Running Tests

The project includes a suite of unit tests and statistical stability checks:
```powershell
python src/tests.py
```

## How It Works

1.  **Harvest Phase**: Agents calculate potential resource gain and generate waste.
2.  **Update Phase**: Agents update internal energy levels. If waste in their cell exceeds their tolerance, they suffer an energy penalty.
3.  **Move & Reproduce Phase**: Agents evaluate neighbors based on a scoring function (Resources, Waste, Distance) and move probabilistically. If energy is sufficient, they reproduce, passing on a mutated genome to their offspring.
