import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
from typing import Dict, List, Any

# Ensure we are in the src directory for imports to work correctly
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Mocking the simulation engine components for standalone execution if needed,
# though the instructions assume grid.py, agent.py, and simulation.py exist.
try:
    from grid import Grid
    from agent import Agent, Genome
    from simulation import Simulation
except ImportError:
    # Fallback for environment issues or if running outside the project structure
    class Grid:
        def __init__(self, w, h):
            self.width, self.height = w, h
            self.resource_a = np.random.rand(w, h)
            self.waste_w = np.random.rand(w, h)
            self.life_occupancy = np.zeros((w, h), dtype=bool)

    class Genome:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
        def to_dict(self): return vars(self)

    class Agent:
        def __init__(self, pos, genome, energy):
            self.position, self.genome, self.energy = pos, genome, energy

    class Simulation:
        def __init__(self, grid, agents):
            self.grid, self.agents = grid, agents
        def tick(self):
            self.grid.resource_a = np.clip(self.grid.resource_a + np.random.uniform(-0.01, 0.01, self.grid.resource_a.shape), 0, 1)
            self.grid.waste_w += np.random.uniform(0, 0.01, self.grid.waste_w.shape)
            # Random movement for dummy
            for a in self.agents:
                a.position = ((a.position[0] + np.random.randint(-1, 2)) % self.grid.width,
                              (a.position[1] + np.random.randint(-1, 2)) % self.grid.height)

# --- State Management ---

def init_session_state():
    """
    Initializes the Streamlit session state to maintain simulation continuity.
    Ensures that simulation objects and history are preserved across reruns.
    """
    if 'simulation' not in st.session_state:
        st.session_state.simulation = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False

# --- Panel 1: Control Center & Hypothesis ---

def render_control_center():
    """
    Renders the Input Panel (Panel 1).
    Collects user hypothesis and simulation parameters.
    Maps:
    - st.text_area -> hypothesis string
    - st.slider -> initial waste threshold
    - st.button -> trigger simulation cycle
    """
    st.sidebar.header("🔬 Control Center")
    
    hypothesis = st.sidebar.text_area(
        "Experiment Hypothesis",
        placeholder="Enter your prediction here...",
        help="Document the expected outcome of this stochastic run."
    )
    
    waste_threshold = st.sidebar.slider(
        "Initial Waste_W Threshold",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Sets the initial toxicity sensitivity for the agent population."
    )
    
    num_steps = st.sidebar.slider("Simulation Steps", 10, 500, 100)
    
    if st.sidebar.button("🔬 Run New Experiment"):
        st.session_state.experiment_running = True
        run_simulation_cycle(waste_threshold, num_steps)

# --- Panel 2: The Observation Field ---

def render_observation_field(grid_data: Dict[str, np.ndarray]):
    """
    Renders the Visual Real-Time View (Panel 2).
    Transforms the raw NumPy grid arrays into a color-mapped visualization.
    
    Data Transformer:
    - grid_data['resource'] (NumPy Array) -> Plotly Heatmap
    - grid_data['occupancy'] (NumPy Array) -> Scatter overlay of agents
    """
    st.subheader("🛰️ Observation Field")
    
    # Create a composite view or use a placeholder for animation
    # In a real scenario, we might use st.image or plotly
    res = grid_data['resource']
    occ = grid_data['occupancy']
    
    # Simple heatmap of resources
    fig = px.imshow(
        res.T, 
        color_continuous_scale='Viridis',
        labels={'color': 'Resource A'},
        title="Resource Density & Agent Occupancy"
    )
    
    # Overlay agents as dots
    y_coords, x_coords = np.where(occ)
    if len(x_coords) > 0:
        fig.add_scatter(x=x_coords, y=y_coords, mode='markers', 
                        marker=dict(color='red', size=4), name='Agents')
    
    st.plotly_chart(fig, use_container_width=True)

# --- Panel 3: KPI & Analysis Dashboard ---

def render_panel_3(history: List[Dict[str, Any]], current_agents: List[Any]):
    """
    Renders the Quantitative Proof Panel (Panel 3).
    Maps historical data to charts and metrics.
    
    Data Transformer:
    - history (List of Dicts) -> Time-series Plotly Chart
    - current_agents (List of Agents) -> Pandas DataFrame for Genome Distribution
    - agent count -> st.metric
    """
    st.divider()
    st.header("📊 KPI & Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    total_agents = len(current_agents)
    avg_energy = np.mean([a.energy for a in current_agents]) if current_agents else 0
    
    col1.metric("Total Agents", total_agents)
    col2.metric("Avg Energy", f"{avg_energy:.2f}")
    col3.metric("Steps Completed", len(history))
    
    # Time-Series Graph
    if history:
        df_history = pd.DataFrame(history)
        fig_ts = px.line(df_history, x=df_history.index, y=['agent_count', 'avg_waste'], 
                         title="Population & Waste Trends Over Time")
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Genome Distribution Table
    st.subheader("🧬 Genome Distribution")
    if current_agents:
        # Extract genome traits into a list of dicts
        genome_data = []
        for a in current_agents:
            # Handle both real and mock Genome objects
            if hasattr(a.genome, '__dict__'):
                d = a.genome.__dict__.copy()
            else:
                d = {'traits': str(a.genome)}
            genome_data.append(d)
        
        df_genome = pd.DataFrame(genome_data)
        st.write("Current population trait averages:")
        st.dataframe(df_genome.describe().loc[['mean', 'std', 'min', 'max']])
        
        st.write("Sample Population Genes:")
        st.dataframe(df_genome.head(10))
    else:
        st.warning("No agents currently active in the simulation.")

# --- The Critical Pipeline ---

def run_simulation_cycle(waste_threshold: float, num_steps: int):
    """
    Orchestrates the data flow between the UI and the Simulation Engine.
    
    Pipeline:
    1. Reads state from Panel 1 (waste_threshold).
    2. Initializes/Resets Grid and Agents.
    3. Loops through simulation ticks.
    4. Updates Panel 2 (Real-time view) and logs history.
    5. Finalizes Panel 3 (KPIs).
    """
    # Setup initial state
    grid_w, grid_h = 50, 50 # Smaller for UI responsiveness
    grid = Grid(grid_w, grid_h)
    
    # Populate grid with some initial resources
    grid.resource_a = np.random.rand(grid_w, grid_h) * 0.5
    
    # Create starter population
    initial_agents = []
    for _ in range(20):
        pos = (np.random.randint(0, grid_w), np.random.randint(0, grid_h))
        # Ensure it matches the Genome schema from agent.py
        genome = Genome(
            movement_speed=np.random.randint(1, 4),
            energy_collection=np.random.uniform(0.5, 1.5),
            waste_tolerance=waste_threshold,
            vision_range=np.random.randint(1, 3),
            w_resource=1.0,
            w_waste=0.5,
            w_distance=0.2
        )
        agent = Agent(pos, genome, energy=1.0)
        initial_agents.append(agent)
        grid.life_occupancy[pos[0], pos[1]] = True
        
    sim = Simulation(grid, initial_agents)
    st.session_state.simulation = sim
    st.session_state.history = []
    
    # Simulation Loop with animated placeholder
    status_text = st.empty()
    progress_bar = st.progress(0)
    observation_placeholder = st.empty()
    
    for t in range(num_steps):
        sim.tick()
        
        # Log History
        current_history = {
            "step": t,
            "agent_count": len(sim.agents),
            "avg_waste": np.mean(sim.grid.waste_w),
            "avg_resource": np.mean(sim.grid.resource_a)
        }
        st.session_state.history.append(current_history)
        
        # Update UI every 5 ticks or so for performance
        if t % 5 == 0 or t == num_steps - 1:
            status_text.text(f"Simulating Step {t}/{num_steps}...")
            progress_bar.progress((t + 1) / num_steps)
            
            with observation_placeholder.container():
                render_observation_field({
                    'resource': sim.grid.resource_a,
                    'occupancy': sim.grid.life_occupancy
                })
            
            # Brief sleep to allow UI to render (mimicking live feed)
            time.sleep(0.01)
            
        if len(sim.agents) == 0:
            st.error("Extinction Event: All agents have died.")
            break
            
    st.session_state.experiment_running = False
    status_text.text("Experiment Complete.")

# --- Main Entry Point ---

def main():
    st.set_page_config(page_title="Metabolic Life Sim", layout="wide")
    st.title("🔬 Metabolic Cellular Automaton Dashboard")
    st.markdown("""
    Welcome to the Central Command Dashboard. This interface allows you to observe 
    the emergent behavior of metabolic agents in a resource-constrained, toxic environment.
    """)
    
    init_session_state()
    render_control_center()
    
    # If no simulation is running, show dummy/last state
    if st.session_state.simulation:
        # If we just finished or are idle, show final stats
        if not st.session_state.experiment_running:
            render_observation_field({
                'resource': st.session_state.simulation.grid.resource_a,
                'occupancy': st.session_state.simulation.grid.life_occupancy
            })
            render_panel_3(st.session_state.history, st.session_state.simulation.agents)
    else:
        st.info("Adjust parameters in the sidebar and click 'Run New Experiment' to start.")
        # Show dummy layout
        render_panel_3([], [])

if __name__ == "__main__":
    main()
