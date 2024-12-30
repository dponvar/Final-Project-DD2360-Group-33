# BFS CUDA Project


This is the final project for the Applied GPU Programming course (DD2360), completed by Daniel Poncelas and Adrian Fernandez. 
The project focuses on improving the CUDA-implemented BFS program from the Rodinia dataset by using unified memory. 
The goal is to apply the concepts learned throughout the course to enhance the performance and efficiency of the program.

This project contains various implementations of the Breadth-First Search (BFS) algorithm in CUDA, including versions using unified memory, 
performance analysis, and memory management.

---

## Project Structure

The project is organized as follows:

rodinia_3.1_DD2360_Group33/

├── bin/

├── common/

├── cuda/

│   └── bfs/

│                 ├── bfs.cu

│                 ├── bfs_managed.cu

│                 ├── bfs_time.cu

│                 ├── bfs_managed_time.cu


│                 ├── bfs_memory.cu

│                 ├── bfs_managed_memory.cu

│                 ├── kernel.cu

│                 ├── kernel2.cu

│                 ├── README.md

│                 ├── result_managed.txt

│                 ├── result.txt   

│                 ├── compare.py          

├── data/

│   └── bfs/

│                 ├── graph1MW_6.txt

│                 ├── graph4096.txt

│                 ├── graph65536.txt

│                 └── inputGen/

├── LICENSE

├── Makefile

└── README.md


# cuda/bfs/
This folder contains the CUDA source files:
- `bfs.cu`: Basic implementation of the BFS algorithm.
- `bfs_managed.cu`: BFS implementation using unified memory.
- `bfs_time.cu`: BFS with execution time measurement.
- `bfs_managed_time.cu`: BFS with unified memory and execution time measurement.
- `bfs_memory.cu`: BFS optimized for memory management.
- `bfs_managed_memory.cu`: BFS with unified memory and memory management analysis.
- `kernel.cu`: Implementation of the first kernel used.
- `kernel2.cu`: Implementation of the second kernel used.
- `README.md`: Information about the folder.
- `result_managed.txt`: Results obtained of the version with Unified Memory.
- `result.txt`: Original results.
- `compare.py`: Script to compare the results to see if they are equivalent 

### 'data/bfs/`
This folder contains input data for testing:
- `graph1MW_6.txt`: Large graph for testing.
- `graph4096.txt` and `graph65536.txt`: Smaller graphs for additional tests.
- `inputGen/`: Data generator (not directly used).

### Other files
- `LICENSE`: Project license.
- `Makefile`: File for automating compilation and execution.
- `README.md`: This documentation file.

---

## Using the Makefile

The `Makefile` simplifies the compilation and execution of the programs.

### Main Commands

1. Compilation  
   To compile all CUDA files, run:  
    `make run`

    This will execute the programs in the following order:

    bfs.cu with graph1MW_6.txt.
    bfs_managed.cu with graph1MW_6.txt.
    bfs_time.cu and bfs_managed_time.cu with all three input files: graph1MW_6.txt, graph4096.txt, and graph65536.txt.
    bfs_memory.cu and bfs_managed_memory.cu with graph1MW_6.txt.

    The output will be displayed in the terminal.

    
    To delete the generated executables and the result file:

    `make clean`

2. Example Output
    
    Compiling cuda/bfs/bfs.cu
   
    Giving permissions to cuda/bfs/bfs
    
    Executing bfs.cu with graph1MW_6.txt
   
    <program output>

    
    Executing bfs_managed.cu with graph1MW_6.txt
   
    <program output>

    ...
    

4. Requirements
    
    CUDA Toolkit: Ensure the CUDA toolkit is installed.
   
    Operating System: Compatible with Linux and systems with CUDA support.
   
    Development Environment: Suitable for local terminals or Google Colab.

5. Additional Notes
    Results are saved in result_managed.txt if configured in the code.
    The project is based on Rodinia 3.1 and includes advanced memory and performance optimizations.
