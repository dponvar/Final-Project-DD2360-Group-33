/***********************************************************************************
  Complete
 ************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

// Define the maximum number of threads per block in CUDA
#define MAX_THREADS_PER_BLOCK 512

// Global variables for the number of nodes and edges in the graph
int no_of_nodes;
int edge_list_size;
FILE *fp; // File pointer for reading input

// Structure to hold node information in the graph
struct Node {
    int starting; // Index where the node's edges start in the edge list
    int no_of_edges; // Number of edges connected to the node
};

#include "kernel.cu"  // Kernel 1 implementation
#include "kernel2.cu" // Kernel 2 implementation

// Function prototype for the main BFS function
void BFSGraph(int argc, char **argv);

// Function to print GPU memory usage at different stages
void printCudaMemoryUsage(const char *stage) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[%s] GPU Memory: Free = %.2f MB, Total = %.2f MB, Used = %.2f MB\n",
           stage, free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
           (total_mem - free_mem) / (1024.0 * 1024.0));
}

// Function to print CPU memory usage at different stages
void printCpuMemoryUsage(const char *stage) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("[%s] CPU Memory: Max Resident Set Size = %.2f MB\n",
           stage, usage.ru_maxrss / 1024.0);
}

////////////////////////////////////////////////////////////////////////////////
// Main Program Entry Point
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    no_of_nodes = 0; // Initialize the number of nodes
    edge_list_size = 0; // Initialize the number of edges
    printf("Initializing BFSGraph...\n");
    BFSGraph(argc, argv); // Call the BFS graph function
    printf("Execution completed.\n");
}

// Function to display usage information
void Usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
// Breadth-First Search (BFS) on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
    char *input_f;

    // Validate command-line arguments
    if (argc != 2) {
        Usage(argc, argv);
        exit(0);
    }

    input_f = argv[1]; // Get the input file name
    printf("Reading File: %s\n", input_f);

    // Open the input file
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0; // Source node for BFS

    // Read the number of nodes in the graph
    fscanf(fp, "%d", &no_of_nodes);
    printf("Number of nodes: %d\n", no_of_nodes);

    // Determine execution configuration: blocks and threads per block
    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int) ceil(no_of_nodes / (double) MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }
    printf("Execution Configuration -> Blocks: %d, Threads per block: %d\n", num_of_blocks, num_of_threads_per_block);

    // Allocate memory for the graph nodes on the host
    Node *h_graph_nodes = (Node *) malloc(sizeof(Node) * no_of_nodes);
    bool *h_graph_mask = (bool *) malloc(sizeof(bool) * no_of_nodes);
    bool *h_updating_graph_mask = (bool *) malloc(sizeof(bool) * no_of_nodes);
    bool *h_graph_visited = (bool *) malloc(sizeof(bool) * no_of_nodes);

    // Read the graph node data from the file
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

    // Read the source node from the file (overwritten to 0 in this implementation)
    fscanf(fp, "%d", &source);
    source = 0;
    printf("Source node: %d\n", source);

    h_graph_mask[source] = true; // Mark the source node in the mask
    h_graph_visited[source] = true; // Mark the source node as visited

    // Read the edge list size
    fscanf(fp, "%d", &edge_list_size);
    printf("Edge list size: %d\n", edge_list_size);

    // Allocate memory for the graph edges on the host
    int id, cost;
    int *h_graph_edges = (int *) malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }
    printf("Graph edges initialized.\n");

    // Close the input file
    if (fp) fclose(fp);
    printf("File reading completed.\n");

    // Print memory usage before device memory allocation
    printCudaMemoryUsage("Before Device Memory Allocation");
    printCpuMemoryUsage("Before Device Memory Allocation");

    // Allocate and copy data to device memory
    Node *d_graph_nodes;
    cudaMalloc((void **) &d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);

    int *d_graph_edges;
    cudaMalloc((void **) &d_graph_edges, sizeof(int) * edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);

    bool *d_graph_mask;
    cudaMalloc((void **) &d_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    bool *d_updating_graph_mask;
    cudaMalloc((void **) &d_updating_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    bool *d_graph_visited;
    cudaMalloc((void **) &d_graph_visited, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    int *h_cost = (int *) malloc(sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++) h_cost[i] = -1;
    h_cost[source] = 0;

    int *d_cost;
    cudaMalloc((void **) &d_cost, sizeof(int) * no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    bool *d_over;
    cudaMalloc((void **) &d_over, sizeof(bool));

    // Print memory usage after device memory allocation
    printCudaMemoryUsage("After Device Memory Allocation");
    printCpuMemoryUsage("After Device Memory Allocation");

    // Kernel execution configuration
    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    int k = 0; // Iteration counter
    printf("Starting BFS traversal.\n");
    bool stop;

    // Main BFS loop executed on the GPU
    do {
        stop = false; // Reset the stop flag
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);

        // Print memory usage before kernel execution
        printCudaMemoryUsage("Before Kernels Execution");
        printCpuMemoryUsage("Before Kernels Execution");

        // Launch Kernel 1
        Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);

        // Launch Kernel 2
        Kernel2<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

        // Copy the stop flag back to the host
        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);

        // Print memory usage after kernel execution
        printCudaMemoryUsage("After Kernels Execution");
        printCpuMemoryUsage("After Kernels Execution");

        k++; // Increment iteration counter
    } while (stop);

    printf("Kernel executed %d times.\n", k);

    // Copy the result back to the host
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    // Write the results to an output file
    FILE *fpo = fopen("result.txt", "w");
    for (int i = 0; i < no_of_nodes; i++) fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
    printf("Results stored in result.txt.\n");

    // Free host and device memory
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

    // Print memory usage after memory cleanup
    printCudaMemoryUsage("After Memory Free");
    printCpuMemoryUsage("After Memory Free");
}
