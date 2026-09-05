# Agentic AI Graph Fraud Agent

A graph-based Agentic AI prototype for detecting suspicious transaction behavior using spectral graph mathematics and random-walk scoring.

## Mathematics
- Graph adjacency matrix
- Degree-normalized Laplacian L = I − D⁻¹ᐟ²AD⁻¹ᐟ²
- Spectral decomposition and low-frequency graph structure
- Personalized random walks
- Structural anomaly scoring

## Agent Workflow
Planner → Graph Builder → Spectral Analyst → Random-Walk Investigator → Fraud Validator → Decision Synthesizer

## Run
```bash
pip install -r requirements.txt
python -m src.main
pytest -q
```

## Structure
`src/` graph agents/tools · `tests/` mathematical tests · `data/` sample graph · `docs/` architecture and math · `notebooks/` experiments · `results/` reproducible notes.

## Interview topics
Spectral graph theory, normalized Laplacian, eigenvectors, random walks, graph anomaly detection, explainability, tool orchestration, and bounded agent loops.
