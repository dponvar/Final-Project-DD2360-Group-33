/*
This program implements a Breadth-First Search (BFS) algorithm on a graph using CUDA for parallel execution. 
The graph is read from an input file, and BFS is performed starting from a specified source node. 
The program makes use of CUDA's unified memory, which allows both the CPU and GPU to access the same memory space, 
simplifying memory management. Additionally, the program measures the execution time of the BFS algorithm using CUDA events, 
performing multiple executions to compute an average execution time. 
Finally, the results, including the BFS traversal cost for each node, are stored in an output file.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes; // Total number of nodes in the graph
int edge_list_size; // Number of edges in the graph
FILE *fp; // File pointer for reading graph data

// Structure to hold node information: starting node index and number of edges for each node
struct Node
{
    int starting;
    int no_of_edges;
};

#include "kernel.cu" // Kernel for BFS processing
#include "kernel2.cu" // Kernel for post-processing after each BFS iteration

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    no_of_nodes = 0; // Initialize number of nodes
    edge_list_size = 0; // Initialize edge list size
    printf("Initializing BFSGraph...\n");

    // Variables for measuring execution time
    cudaEvent_t start, stop; // CUDA event objects to track execution time
    float elapsed_time_ms; // Variable to store elapsed time for each execution
    float total_time_ms = 0.0f; // Accumulator to calculate the average execution time
    const int NUM_EXECUTIONS = 10; // Number of times to run BFS for averaging the time

    // Create CUDA events to record start and stop times for execution
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Loop for multiple executions to compute average runtime
    for (int i = 0; i < NUM_EXECUTIONS; i++) {
        // Record the start time
        cudaEventRecord(start, 0);

        // Execute BFSGraph to process the graph (defined below)
        BFSGraph(argc, argv);

        // Record the stop time after the function execution
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); // Ensure the stop event is fully completed
        cudaEventElapsedTime(&elapsed_time_ms, start, stop); // Calculate the elapsed time

        // Print the elapsed time for the current execution
        printf("Execution %d time: %.3f ms\n", i + 1, elapsed_time_ms);

        // Add the execution time to the total time
        total_time_ms += elapsed_time_ms;
    }

    // Calculate the average time after multiple executions
    float average_time_ms = total_time_ms / NUM_EXECUTIONS;
    printf("Average execution time after %d runs: %.3f ms\n", NUM_EXECUTIONS, average_time_ms);

    // Clean up the CUDA event objects
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Execution completed.\n");
}

// Function to display usage information if the program is not called correctly
void Usage(int argc, char**argv) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) 
{
    char *input_f;
    if (argc != 2) {
        Usage(argc, argv); // If incorrect arguments, show usage and exit
        exit(0);
    }
    
    input_f = argv[1]; // Get the graph input file name from arguments
    printf("Reading File: %s\n", input_f);

    // Open the input file for reading graph data
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0; // Initialize source node for BFS (default 0)
    // Read the number of nodes from the file
    if (fscanf(fp, "%d", &no_of_nodes) != 1) {
        fprintf(stderr, "Error reading number of nodes.\n");
        fclose(fp); // Close file in case of error
        exit(EXIT_FAILURE);
    }
    printf("Number of nodes: %d\n", no_of_nodes);

    // Calculate the number of blocks and threads per block for optimal execution
    int num_of_blocks = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    int num_of_threads_per_block = (no_of_nodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : no_of_nodes;
    printf("Execution Configuration -> Blocks: %d, Threads per block: %d\n", num_of_blocks, num_of_threads_per_block);

    // Allocate unified memory for graph structures using CUDA's unified memory model
    Node* d_graph_nodes; // Array of nodes (Node struct)
    bool *d_graph_mask, *d_updating_graph_mask, *d_graph_visited, *d_over; // Graph traversal masks
    int* d_graph_edges, *d_cost; // Graph edges and node costs

    // Check memory allocation success for the unified memory arrays
    if (cudaMallocManaged(&d_graph_nodes, sizeof(Node) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_graph_mask, sizeof(bool) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_updating_graph_mask, sizeof(bool) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_graph_visited, sizeof(bool) * no_of_nodes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory for graph structures.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Initialize the graph data (nodes and edges) from the input file
    int start, edgeno;
    for (unsigned int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
            fprintf(stderr, "Error reading node information.\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        d_graph_nodes[i].starting = start;
        d_graph_nodes[i].no_of_edges = edgeno;
        d_graph_mask[i] = false; // Mark nodes as unvisited
        d_updating_graph_mask[i] = false;
        d_graph_visited[i] = false;
    }
    printf("Graph nodes initialized.\n");

    // Read the source node for BFS traversal
    fscanf(fp, "%d", &source);
    source = 0; // For simplicity, set the source to 0
    printf("Source node: %d\n", source);

    // Mark the source node as visited
    d_graph_mask[source] = true;
    d_graph_visited[source] = true;

    // Read the size of the edge list from the file
    if (fscanf(fp, "%d", &edge_list_size) != 1) {
        fprintf(stderr, "Error reading edge list size.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    printf("Edge list size: %d\n", edge_list_size);

    // Allocate memory for edges and initialize from file
    cudaMallocManaged(&d_graph_edges, sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        int id, cost;
        if (fscanf(fp, "%d %d", &id, &cost) != 2) {
            fprintf(stderr, "Error reading edge information.\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        d_graph_edges[i] = id; // Store the edge information (id and cost)
    }
    printf("Graph edges initialized.\n");

    // Close the file after reading all necessary data
    if (fp) fclose(fp);
    printf("File reading completed.\n");

    // Allocate memory for the cost array and the "over" flag (used to stop BFS traversal)
    cudaMallocManaged(&d_cost, sizeof(int) * no_of_nodes);
    cudaMallocManaged(&d_over, sizeof(bool));
    for (int i = 0; i < no_of_nodes; i++) d_cost[i] = -1; // Initialize costs to -1
    d_cost[source] = 0; // Cost of the source node is 0
    printf("Memory allocated and copied to GPU.\n");

    // Define grid and thread configuration for CUDA kernel launch
    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    int k = 0; // BFS iteration counter
    printf("Starting BFS traversal.\n");
    bool *stop; // Flag to indicate whether the BFS traversal is complete
    cudaMallocManaged(&stop, sizeof(bool)); // Allocate unified memory for the stop flag

    // BFS loop: Keep processing the graph until there are no more nodes to visit
    do {
        *stop = false; // Reset stop condition for each iteration
        printf("Kernel iteration %d\n", k);
        
        // Launch the first kernel to process the BFS traversal
        Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, 
                                  d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
        
        // Launch the second kernel to update the visited nodes and check for stopping condition
        Kernel2<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask, 
                                   d_graph_visited, stop, no_of_nodes);
        
        // Synchronize the device to ensure the kernels are finished
        cudaDeviceSynchronize();
        
        k++; // Increment the iteration counter
    } while (*stop); // Continue until no nodes are left to process

    printf("Kernel executed %d times.\n", k);

    // Free allocated memory
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_graph_edges);
    cudaFree(d_cost);
    cudaFree(d_over);
    cudaFree(stop);
}
