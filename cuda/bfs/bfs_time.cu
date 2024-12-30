/*
This program implements a Breadth-First Search (BFS) algorithm on a graph using CUDA for parallel processing. 
The graph is read from an input file, and the BFS traversal starts from a specified source node. 
Key features include memory allocation on both host (CPU) and device (GPU), execution of CUDA kernels to perform the BFS, and 
calculating the average runtime over multiple executions for performance evaluation. The results are stored in an output file.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512 // Maximum number of threads per CUDA block

int no_of_nodes;         // Number of nodes in the graph
int edge_list_size;      // Total number of edges in the graph
FILE *fp;                // File pointer to read the graph data

// Structure to represent each node in the graph
struct Node {
    int starting;       // Starting index in the edge list for this node
    int no_of_edges;    // Number of edges originating from this node
};

#include "kernel.cu"   // Kernel 1: Performs the main BFS computation
#include "kernel2.cu"  // Kernel 2: Updates graph mask and checks for stopping conditions

void BFSGraph(int argc, char** argv); // Function declaration for BFS implementation

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    no_of_nodes = 0;
    edge_list_size = 0;
    printf("Initializing BFSGraph...\n");

    // Variables to measure execution time
    cudaEvent_t start, stop;
    float elapsed_time_ms;
    float total_time_ms = 0.0f; // Accumulator for average time calculation
    const int NUM_EXECUTIONS = 10; // Number of executions for averaging

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < NUM_EXECUTIONS; i++) {
        // Record start time
        cudaEventRecord(start, 0);

        // Call BFSGraph function
        BFSGraph(argc, argv);

        // Record end time and calculate elapsed time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        // Print the time for the current execution
        printf("Execution %d time: %.3f ms\n", i + 1, elapsed_time_ms);

        // Accumulate total time
        total_time_ms += elapsed_time_ms;
    }

    // Calculate average execution time
    float average_time_ms = total_time_ms / NUM_EXECUTIONS;
    printf("Average execution time after %d runs: %.3f ms\n", NUM_EXECUTIONS, average_time_ms);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Execution completed.\n");
}

// Prints usage information if the program is called with incorrect arguments
void Usage(int argc, char**argv) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) {
    char *input_f;

    // Validate command-line arguments
    if (argc != 2) {
        Usage(argc, argv);
        exit(0);
    }

    input_f = argv[1];
    printf("Reading File: %s\n", input_f);

    // Open input file
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0; // Default source node

    // Read the number of nodes in the graph
    fscanf(fp, "%d", &no_of_nodes);
    printf("Number of nodes: %d\n", no_of_nodes);

    // Determine CUDA grid and block dimensions
    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;
    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }
    printf("Execution Configuration -> Blocks: %d, Threads per block: %d\n", num_of_blocks, num_of_threads_per_block);

    // Allocate host memory for graph data
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node) * no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool) * no_of_nodes);
    bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool) * no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool) * no_of_nodes);

    // Read node information from the file
    int start, edgeno;
    for (unsigned int i = 0; i < no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }
    printf("Graph nodes initialized.\n");

    // Read the source node from the file
    fscanf(fp, "%d", &source);
    printf("Source node: %d\n", source);
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    // Read edge list size and edge data
    fscanf(fp, "%d", &edge_list_size);
    printf("Edge list size: %d\n", edge_list_size);
    int id, cost;
    int* h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }
    printf("Graph edges initialized.\n");

    if (fp) fclose(fp); // Close the input file
    printf("File reading completed.\n");

    // Allocate and copy data to device memory
    Node* d_graph_nodes;
    cudaMalloc((void**) &d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);

    int* d_graph_edges;
    cudaMalloc((void**) &d_graph_edges, sizeof(int) * edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);

    bool* d_graph_mask;
    cudaMalloc((void**) &d_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    bool* d_updating_graph_mask;
    cudaMalloc((void**) &d_updating_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    bool* d_graph_visited;
    cudaMalloc((void**) &d_graph_visited, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    int* h_cost = (int*) malloc(sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++) h_cost[i] = -1;
    h_cost[source] = 0;

    int* d_cost;
    cudaMalloc((void**) &d_cost, sizeof(int) * no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    bool *d_over;
    cudaMalloc((void**) &d_over, sizeof(bool));
    printf("Memory allocated and copied to GPU.\n");

    dim3 grid(num_of_blocks, 1, 1); // CUDA grid configuration
    dim3 threads(num_of_threads_per_block, 1, 1); // CUDA block configuration

    int k = 0;
    printf("Starting BFS traversal.\n");
    bool stop;
    do {
        stop = false;
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
        printf("Kernel iteration %d\n", k);

        // Execute CUDA kernels
        Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
        Kernel2<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

        // Copy stop condition back to host
        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
    } while (stop);

    printf("Kernel executed %d times.\n", k);

    // Copy results from device to host
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);
    FILE *fpo = fopen("result.txt", "w");
    for (int i = 0; i < no_of_nodes; i++) fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
    printf("Results stored in result.txt.\n");

    // Free allocated memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
}
