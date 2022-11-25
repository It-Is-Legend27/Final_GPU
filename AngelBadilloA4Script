#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J mysimplejob           # Job name
#SBATCH -o mysimplejob.%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx		             # Queue name (EC: Options: gtx, v100, p100)
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 01:00:00              # Run time (hh:mm:ss) - 1.0 hours

#SBATCH -A CMPS-5433-MWSU        # Specify allocation to charge against

#============================
# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.
module load hdf5
module load cuda/10.1   #EC updated 02/10/2020

#============================
# COMPILE YOUR CODE HERE
#EC updated 02/10/2020: OLD Stuff nvcc -arch=compute_35 -code=sm_35 helloWorld.cu -o a.out
# nvcc is your compiler
# helloWorld is the name of the file that has your code
# -o is a compilation flag
# a.out is the name of your executable

nvcc AngelBadilloA4.cu -o a.out

#============================
# LAUNCH YOUR EXECUTABLE
# Launch the executable named "a.out"
./a.out
