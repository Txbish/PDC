import Pkg;
# Pkg.add("TimerOutputs")
# Pkg.add("QXTools")
# Pkg.add("QXGraphDecompositions")
# Pkg.add("QXZoo")
# Pkg.add("DataStructures")
# Pkg.add("QXTns")
# Pkg.add("NDTensors")
# Pkg.add("ITensors")
# Pkg.add("LightGraphs")
# Pkg.add("PyCall")
# Pkg.add("MPI")


# Using required modules
using MPI
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

# Load custom functions from the folder src
include("./src/funcions_article.jl");


@info("What type of circuit do you want to create? (GHZ, QFT, RQC)\n\n")
circuit_type_input = readline()
circuit_type = uppercase(strip(circuit_type_input))

@info("How many qubits do you want for the circuit (n)?\n\n")
N_input = readline()
n = parse(Int, N_input)

local cct # Use local to ensure cct is available in this scope

if circuit_type == "GHZ"
    # Create GHZ circuit
    cct = create_ghz_circuit(n)
    @info("GHZ circuit with $(n) qubits created\n\n")
elseif circuit_type == "QFT"
    # Create QFT circuit
    cct = create_qft_circuit_bis(n) # Assuming create_qft_circuit_bis is defined in funcions_article.jl
    @info("QFT circuit with $(n) qubits created\n\n")
elseif circuit_type == "RQC"

    @info("How deep do you want for the RQC circuit (d)?\n\n")
    rqc_d_input = readline()
    d_rqc = parse(Int, rqc_d_input)

    @info("Give us a seed as a positive number, please (seed)?\n\n")
    rqc_seed_input = readline()
    seed_rqc = parse(Int, rqc_seed_input)

    # Create RQC circuit
    cct = create_rqc_circuit(n, n, d_rqc, seed_rqc, final_h=true) # Assuming this is the correct signature
    @info("RQC circuit $(n)x$(n) with depth $(d_rqc) and seed $(seed_rqc) created\n\n")
else
    @error("Invalid circuit type entered. Please choose 'GHZ', 'QFT', or 'RQC'.")
    exit() # Exit if the circuit type is not recognized
end

tnc = convert_to_tnc(cct)  # Convert the circuit into a tensor network circuit

s1 = Calcul_GN_Sequencial(cct, true)
println("Sequential contraction result: ", s1)

