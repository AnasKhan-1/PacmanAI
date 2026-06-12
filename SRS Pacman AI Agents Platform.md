# Software Requirements Specification

## Pacman AI Agents Platform

Version 1.0 | Prepared for: AI Project | Status: Draft

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for a Pacman game built from scratch in Python, together with a set of swappable AI agents (search, adversarial, and reinforcement learning) and an evaluation framework that compares them. It is intended for the project developer, supervisor, and reviewers.

### 1.2 Scope

The system consists of:

- A deterministic, grid-based Pacman game engine separated from its rendering layer.
- A playable graphical interface built with Pygame.
- A common agent interface so any AI technique can control Pacman.
- Implementations of random, BFS, A\*, minimax, expectimax, Q-learning, approximate Q-learning, and (optionally) DQN agents.
- A headless experiment runner that produces comparison tables and charts.

Out of scope: online multiplayer, mobile deployment, sound design, and pixel-perfect replication of the original arcade game.

### 1.3 Definitions and Acronyms

| Term | Meaning |
| --- | --- |
| SRS | Software Requirements Specification |
| Agent | A program that selects Pacman's next action from the game state |
| BFS / A\* | Breadth-First Search / A-star pathfinding algorithms |
| Minimax / Expectimax | Adversarial search algorithms |
| RL | Reinforcement Learning |
| DQN | Deep Q-Network |
| Headless | Running the game without graphics |
| Episode | One full game from start to win or loss |

### 1.4 References

- IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.
- UC Berkeley CS188 Pacman projects (as design inspiration).
- Sutton and Barto, Reinforcement Learning: An Introduction.

### 1.5 Overview

Section 2 describes the product in general terms. Section 3 lists specific requirements. Section 4 covers other requirements, and Section 5 gives the development plan and acceptance criteria.

## 2. Overall Description

### 2.1 Product Perspective

The product is a standalone desktop application and command-line toolkit. It has three layers: the game engine (rules and state), the agents (decision logic), and the presentation and experiment layer (Pygame UI, logging, plotting). The engine has no dependency on the graphics library.

### 2.2 Product Functions

- Simulate Pacman on a configurable maze with pellets, power pellets, walls, and ghosts.
- Let a human play with the keyboard.
- Let an AI agent play through a standard interface.
- Run many games quickly without graphics and record statistics.
- Train RL agents and save and load learned parameters.
- Generate comparison tables and charts across agents.

### 2.3 User Classes

| User | Description | Needs |
| --- | --- | --- |
| Developer/Student | Builds and extends agents | Clean agent API, reproducible runs |
| Reviewer/Supervisor | Evaluates the project | Clear results, readable reports, demo mode |
| Player | Plays the game manually | Responsive controls, clear UI |

### 2.4 Operating Environment

- Windows, macOS, or Linux.
- Python 3.10 or later.
- Libraries: Pygame, NumPy, Matplotlib; PyTorch (only for DQN).
- No internet connection required at runtime.

### 2.5 Design and Implementation Constraints

- Game logic must be independent of rendering.
- The game advances in discrete, turn-based steps.
- All randomness must be controlled by a seed.
- Code must be modular, documented, and version controlled.

### 2.6 Assumptions and Dependencies

- The developer has working knowledge of Python, classes, and basic data structures.
- A machine capable of running small neural network training on CPU is available.
- Third-party libraries remain compatible with the chosen Python version.

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 Game Engine

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | The system shall load a maze from a text file where characters represent walls, pellets, power pellets, Pacman start, and ghost starts. | High |
| FR-2 | The system shall represent the game as an immutable or copyable state object containing positions, pellets, score, lives, and ghost modes. | High |
| FR-3 | The system shall provide a function that returns the legal actions (North, South, East, West, Stop) for any agent in any state. | High |
| FR-4 | The system shall provide a successor function that, given a state and an action, returns the next state without modifying the original. | High |
| FR-5 | Pacman shall eat a pellet when entering its cell, increasing the score by a configurable amount. | High |
| FR-6 | Eating a power pellet shall make ghosts vulnerable for a configurable number of steps. | Medium |
| FR-7 | A collision with a non-vulnerable ghost shall cost Pacman a life or end the game; a collision with a vulnerable ghost shall send the ghost back to its start and award bonus points. | High |
| FR-8 | The game shall end with a win when all pellets are eaten and with a loss when lives reach zero or a step limit is exceeded. | High |
| FR-9 | The engine shall support a configurable number of ghosts (0 to 4). | Medium |
| FR-10 | The engine shall run without any graphics dependency (headless mode). | High |

#### 3.1.2 Ghost Behavior

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-11 | The system shall provide a random ghost that picks a legal action uniformly. | High |
| FR-12 | The system shall provide a chase ghost that moves toward Pacman using shortest-path search. | High |
| FR-13 | The system shall provide an optional scatter/chase mode switch on a timer. | Low |
| FR-14 | Ghost behavior shall be selectable by configuration. | Medium |

#### 3.1.3 User Interface

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-15 | The system shall render the maze, Pacman, ghosts, pellets, score, and lives using Pygame. | High |
| FR-16 | A human player shall control Pacman with the arrow keys or WASD. | High |
| FR-17 | The UI shall allow selecting an AI agent to watch play, with adjustable playback speed. | Medium |
| FR-18 | The UI shall allow pausing, restarting, and quitting. | Medium |
| FR-19 | The UI shall show the current agent name and live statistics. | Low |

#### 3.1.4 Agent Framework

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-20 | The system shall define an agent interface with a method that takes a game state and returns an action. | High |
| FR-21 | Agents shall be interchangeable through configuration without changes to engine code. | High |
| FR-22 | The system shall provide a random agent as a baseline. | High |
| FR-23 | The system shall provide BFS and A\* agents that plan a path to the nearest pellet (A\* using Manhattan or maze distance as the heuristic). | High |
| FR-24 | The system shall provide minimax with alpha-beta pruning and a configurable search depth. | High |
| FR-25 | The system shall provide an expectimax agent that models ghosts as random. | High |
| FR-26 | The system shall provide a custom evaluation function usable by the adversarial agents, based on features such as distance to nearest pellet, distance to ghosts, and remaining pellets. | High |
| FR-27 | The system shall provide a tabular Q-learning agent with configurable learning rate, discount factor, and exploration rate. | High |
| FR-28 | The system shall provide an approximate Q-learning agent using a feature-based linear function approximator. | High |
| FR-29 | The system shall optionally provide a DQN agent with experience replay and a target network. | Low |
| FR-30 | RL agents shall support saving and loading learned weights or tables to disk. | Medium |

#### 3.1.5 Training and Experiments

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-31 | The system shall provide a command-line runner that executes N episodes for a chosen agent, maze, and seed. | High |
| FR-32 | The runner shall record, per episode: win or loss, final score, steps survived, pellets eaten, and runtime. | High |
| FR-33 | The runner shall support a training mode (learning on) and an evaluation mode (learning off, exploration off). | High |
| FR-34 | The system shall export results to CSV. | High |
| FR-35 | The system shall generate charts including win rate, average score, survival time, and RL learning curves. | Medium |
| FR-36 | The system shall produce a side-by-side comparison table of all agents on the same mazes and seeds. | High |
| FR-37 | The system shall allow configuration through a file or command-line arguments rather than code edits. | Medium |

### 3.2 External Interface Requirements

#### 3.2.1 User Interfaces

A single Pygame window shows the game board and a status bar. A command-line interface exposes the experiment runner.

#### 3.2.2 Hardware Interfaces

Keyboard input for manual play. Optional GPU for DQN training; CPU must be sufficient for all other features.

#### 3.2.3 Software Interfaces

- Pygame for rendering and input.
- NumPy for numerical operations.
- Matplotlib for charts.
- PyTorch for DQN (optional).

#### 3.2.4 File Interfaces

- Maze files: plain text grids.
- Config files: JSON or YAML.
- Result files: CSV.
- Model files: pickle, JSON, or PyTorch checkpoint.

### 3.3 Non-Functional Requirements

#### 3.3.1 Performance

| ID | Requirement |
| --- | --- |
| NFR-1 | Headless simulation shall achieve at least 1,000 game steps per second on a standard laptop for a small maze with non-search agents. |
| NFR-2 | An expectimax or minimax agent at depth 2 shall choose an action in under 100 ms on a 20x11 maze. |
| NFR-3 | The UI shall run at a stable 30 frames per second or better. |

#### 3.3.2 Reliability and Correctness

| ID | Requirement |
| --- | --- |
| NFR-4 | Given the same seed, maze, and agent, a run shall produce identical results. |
| NFR-5 | The engine shall never allow movement into walls or outside the grid. |
| NFR-6 | Invalid agent actions shall be handled gracefully (rejected or replaced with Stop) and logged. |

#### 3.3.3 Usability

| ID | Requirement |
| --- | --- |
| NFR-7 | A new user shall be able to start a game or run an experiment within five minutes using the README. |
| NFR-8 | Error messages shall state what went wrong and how to fix it. |

#### 3.3.4 Maintainability and Extensibility

| ID | Requirement |
| --- | --- |
| NFR-9 | Adding a new agent shall require only one new class implementing the agent interface and one registry entry. |
| NFR-10 | Core engine modules shall have unit tests covering movement, collisions, scoring, and win/loss conditions, with at least 80 percent coverage of the engine. |
| NFR-11 | Code shall follow PEP 8 and include docstrings for public classes and functions. |

#### 3.3.5 Portability

| ID | Requirement |
| --- | --- |
| NFR-12 | The system shall run unchanged on Windows, macOS, and Linux. |

### 3.4 Data Requirements

- **State representation:** Pacman position, ghost positions and modes, pellet set, power pellet set, score, lives, step count.
- **RL reward design (configurable defaults):** small penalty per step, positive reward per pellet, larger reward for eating a vulnerable ghost, large positive reward for winning, large negative reward for dying.
- **Feature vector for approximate Q-learning:** distance to nearest pellet, distance to nearest ghost, number of ghosts within a small radius, whether the action eats a pellet, and a bias term.

## 4. Other Requirements

### 4.1 Legal and Licensing

All code is original or uses permissively licensed libraries. Graphics and names should be original or clearly labeled as Pacman-style, since the Pac-Man name and artwork are trademarked. Use custom sprites or simple shapes.

### 4.2 Documentation

- README with installation and usage instructions.
- Architecture diagram of engine, agents, and UI.
- Report describing each algorithm, parameters used, and results.

### 4.3 Risks and Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| RL agent fails to learn | High | Start with a tiny maze, simplify rewards, use feature-based Q-learning before DQN |
| Rendering and logic become entangled | Medium | Enforce the engine/UI separation from day one |
| Search agents too slow | Medium | Alpha-beta pruning, limited depth, cheap evaluation function |
| Scope creep | High | Treat DQN and scatter/chase modes as stretch goals |
| Non-reproducible results | Medium | Seed all random sources and log configurations |

## 5. Development Plan and Acceptance

### 5.1 Milestones

| Phase | Deliverable | Related Requirements |
| --- | --- | --- |
| 1 | Game engine, maze loader, random agent, unit tests | FR-1 to FR-10, FR-20 to FR-22 |
| 2 | Pygame UI and human play | FR-15 to FR-19 |
| 3 | Ghost AI and A\* agent | FR-11 to FR-14, FR-23 |
| 4 | Minimax and expectimax | FR-24 to FR-26 |
| 5 | Q-learning and approximate Q-learning | FR-27 to FR-28, FR-30 |
| 6 | Experiment runner, CSV, charts, comparison | FR-31 to FR-37 |
| 7 | (Stretch) DQN | FR-29 |
| 8 | Final report and demo | Section 4.2 |

### 5.2 Acceptance Criteria

- All High-priority functional requirements are implemented and demonstrated.
- The game is playable by a human and watchable with any implemented agent.
- Unit tests pass and meet the coverage target.
- Runs are reproducible with a fixed seed.
- On the same maze and seeds, the following ordering of average score is demonstrated, or a justified explanation is given if it is not observed: random is lower than BFS/A\*, which is lower than or comparable to expectimax and trained approximate Q-learning.
- A final comparison table and charts are included in the report.

## Appendix A: Suggested Project Structure

```
pacman_ai/
  engine/        # state, rules, maze loader, ghosts
  agents/        # base, random, search, adversarial, rl
  ui/            # pygame renderer, input handling
  experiments/   # runner, config, metrics, plotting
  mazes/         # text layouts
  tests/
  README.md
```

## Appendix B: Revision History

| Version | Date | Description |
| --- | --- | --- |
| 1.0 | 2026-10-18 | Initial draft |
