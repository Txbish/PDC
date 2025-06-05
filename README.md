

# Quantum Circuit Simulation using Tensor Networks

This repository contains the implementation of a **Quantum Circuit Simulator** built using **Tensor Networks**, developed as a part of the **Parallel and Distributed Computing (PDC)** course project. It extends and parallelizes the work from [Multistage_contraction](https://github.com/alfred-miquel/Multistage_contraction) by introducing **distributed and multithreaded execution using MPI, OpenMP, and native Julia threads**, along with **graph partitioning using METIS**.

---

## 🧠 Project Overview

Quantum circuits can be naturally represented as tensor networks. Efficient contraction of these networks is critical for scalable quantum simulation. Our simulator decomposes this problem into three structured and parallelizable phases:

### 🔹 Phase 1: Community Detection
- Analyzes the tensor network to identify **communities** (groups of tightly connected tensors).
- Uses **METIS** for graph partitioning to minimize inter-community connections.

### 🔹 Phase 2: Community Contraction
- Contracts tensors **within each community in parallel**, using:
  - **MPI** for distributed processes
  - **OpenMP** and **Julia threads** for multithreaded performance
- Reduces the network size while preserving global structure.

### 🔹 Phase 3: Final Merging and Contraction
- Merges the contracted communities into a final tensor network.
- Performs a **global contraction** with coordinated parallelism to obtain the simulation result.

---

## 🧪 Features

- ✅ **Community-aware optimization** to speed up tensor contractions
- ⚡ **MPI**, **OpenMP**, and **Julia threads** used for parallelism
- 🧩 **METIS**-based graph partitioning for balanced and efficient task allocation
- 🛠️ Modular design for individual control over each simulation phase
- 🚀 Built in **Julia** for high-performance scientific computing

---


## 🚀 Getting Started

### Requirements

- Julia 1.8+
- System Dependencies:
  - MPI (e.g., `OpenMPI`)
  - METIS
- Julia Packages:
  - `LightGraphs.jl`
  - `TensorOperations.jl`
  - `JSON.jl`
  - `Distributed.jl`
  - `Metis.jl`
  - `MPI.jl`
  - `ThreadsX.jl` (optional)

### Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/txbish/quantum-tensor-simulator.git
cd quantum-tensor-simulator
julia
] activate .
] instantiate
````

### Run Simulation

**Parallel Run (Recommended):**

```bash
mpiexec -n 8 julia parallel.jl 
```

**Serial Run (Baseline):**

```bash
julia -t auto serial.jl 
```

---

## 📈 Performance & Scalability

Our parallel simulator is designed to take advantage of:

* **Inter-node communication** using MPI
* **Intra-node multithreading** using OpenMP and Julia threads
* **Graph partitioning** with METIS for load balancing

This allows scalable simulation of larger circuits with better contraction time and resource utilization.

---

## 📚 Attribution

> This work builds on and extends [Multistage\_contraction](https://github.com/alfred-miquel/Multistage_contraction) by Alfred Miquel. Original concepts and serial structure are adapted with added parallelism and system-level optimization.

---

## 👨‍💻 Contributors

* Tabish Noman Khan
* Arqam Zia
* Muhammad Danish Haroon

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or collaboration:
**LinkedIn:** [txbish](https://www.linkedin.com/in/txbish)

