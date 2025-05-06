# Add necessary packages
# import Pkg; 
# Pkg.add("QXTools")
# Pkg.add("QXGraphDecompositions")
# Pkg.add("QXZoo")
# Pkg.add("TimerOutputs")
# Pkg.add("DataStructures")
# Pkg.add("QXTns")
# Pkg.add("NDTensors")
# Pkg.add("ITensors")
# Pkg.add("LightGraphs")
# Pkg.add("PyCall")
# Pkg.add("MPI")

# Using required modules
using QXTools
using QXTns
using QXZoo
using PyCall
using QXGraphDecompositions
using LightGraphs
using DataStructures
using TimerOutputs
using ITensors
using LinearAlgebra
using NDTensors
using MPI

# Load custom functions from the folder src
include("./src/funcions_article.jl");

# Create a GHZ circuit with 10 qubits
circuit = create_ghz_circuit(999)

# Convert the circuit to a tensor network circuit (TNC)
tnc = convert_to_tnc(circuit)

# Configure the contraction algorithm
num_communities = 8  # Number of communities for the multistage algorithm
input_state = "0"^999  # All qubits initialized to 0
output_state = "1"^999 # Target output state

# Run the ComPar algorithm using multicore CPU
result = ComParCPU_MPI(circuit, input_state, output_state, num_communities;
    timings=true, decompose=true)
