/*
This program implements a parallelized Breadth-First Search (BFS) algorithm on a graph using CUDA. 
It leverages unified memory, allowing both the CPU and GPU to access the same memory space. 
The program processes a graph, which is represented by nodes and edges, and performs BFS starting from a given source node. 
The graph data, such as node connections and the edges, is read from an input file. 
During the BFS execution, the program continuously updates node visitation states and computes the cost (distance) for each node. 
The memory usage of both the CPU and GPU is tracked before and after the kernel executions. 
After the BFS traversal is complete, the results are written to an output file.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_THREADS_PER_BLOCK 512  // Maximum number of threads per block for kernel execution

// Structure to hold information for each node in the graph
struct Node {
    int starting; // Starting index for the edges of the node
    int no_of_edges; // Number of edges for the node
};

#include "kernel.cu" // Kernel for BFS processing
#include "kernel2.cu" // Kernel for post-processing after each BFS iteration

// Function to print the memory usage of the GPU during execution
void printCudaMemoryUsage(const char *stage) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem); // Retrieve the available and total GPU memory
    printf("[%s] GPU Memory: Free = %.2f MB, Total = %.2f MB, Used = %.2f MB\n",
           stage, free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
           (total_mem - free_mem) / (1024.0 * 1024.0));
}

// Function to print the memory usage of the CPU during execution
void printCpuMemoryUsage(const char *stage) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);  // Get CPU memory usage
    printf("[%s] CPU Memory: Max Resident Set Size = %.2f MB\n",
           stage, usage.ru_maxrss / 1024.0); // Max memory used by the program
}



int main(int argc, char **argv) {
    // Validate input arguments
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return -1;
    }

    // Open the graph input file
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        printf("Error reading file %s\n", argv[1]);
        return -1;
    }

    // Read the number of nodes in the graph
    int no_of_nodes, edge_list_size, source;
    fscanf(fp, "%d", &no_of_nodes);

    printf("Number of nodes: %d\n", no_of_nodes);

    // Allocate unified memory for graph nodes and flags
    Node *graph_nodes;
    cudaMallocManaged(&graph_nodes, sizeof(Node) * no_of_nodes);

    bool *graph_mask, *updating_graph_mask, *graph_visited;
    cudaMallocManaged(&graph_mask, sizeof(bool) * no_of_nodes);
    cudaMallocManaged(&updating_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMallocManaged(&graph_visited, sizeof(bool) * no_of_nodes);

    // Initialize graph node data from the input file
    for (int i = 0; i < no_of_nodes; i++) {
        int start, edgeno;
        fscanf(fp, "%d %d", &start, &edgeno);
        graph_nodes[i].starting = start;
        graph_nodes[i].no_of_edges = edgeno;
        graph_mask[i] = false;
        updating_graph_mask[i] = false;
        graph_visited[i] = false;
    }

    // Read the source node and the edge list size
    fscanf(fp, "%d", &source);
    fscanf(fp, "%d", &edge_list_size);

    // Allocate memory for graph edges
    int *graph_edges;
    cudaMallocManaged(&graph_edges, sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d", &graph_edges[i]);
    }

    fclose(fp);  // Close the input file

    printf("Graph data initialized.\n");

    // Initialize BFS starting point (source node)
    graph_mask[source] = true;
    graph_visited[source] = true;

    // Initialize the cost array with -1 (indicating unvisited nodes)
    int *cost;
    cudaMallocManaged(&cost, sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++) cost[i] = -1;
    cost[source] = 0;  // Set the cost of the source node to 0

    // Print memory usage before kernel execution
    printCudaMemoryUsage("Before Kernel Execution");
    printCpuMemoryUsage("Before Kernel Execution");

    // Configure the number of blocks and threads per block for kernel execution
    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;
    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int) ceil(no_of_nodes / (double) MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    // Allocate memory for the "over" flag
    bool *over;
    cudaMallocManaged(&over, sizeof(bool));

    printf("Starting BFS traversal...\n");

    // BFS loop: Keep processing the graph until there are no more nodes to visit
    int k = 0;
    do {
        *over = false;  // Reset the "over" flag to false at the start of each iteration

        // Execute Kernel to perform BFS traversal on the graph
        Kernel<<<grid, threads>>>(graph_nodes, graph_edges, graph_mask,
                                  updating_graph_mask, graph_visited, cost, no_of_nodes);
        cudaDeviceSynchronize();  // Synchronize device to ensure the kernel finishes

        // Execute Kernel2 to update visited nodes and check if BFS should continue
        Kernel2<<<grid, threads>>>(graph_mask, updating_graph_mask, graph_visited,
                                   over, no_of_nodes);
        cudaDeviceSynchronize();  // Synchronize device

        k++;  // Increment the iteration counter
        // Print memory usage after each kernel execution
        printCudaMemoryUsage("After Kernel Execution");
        printCpuMemoryUsage("After Kernel Execution");
    } while (*over);  // Continue while there are still nodes to process

    printf("Kernel executed %d times.\n", k);

    // Store the result in an output file
    FILE *fpo = fopen("result.txt", "w");
    for (int i = 0; i < no_of_nodes; i++) fprintf(fpo, "%d) cost:%d\n", i, cost[i]);
    fclose(fpo);  // Close the output file
    printf("Results stored in result.txt.\n");

    // Free all allocated memory
    cudaFree(graph_nodes);
    cudaFree(graph_edges);
    cudaFree(graph_mask);
    cudaFree(updating_graph_mask);
    cudaFree(graph_visited);
    cudaFree(cost);
    cudaFree(over);

    // Print memory usage after freeing resources
    printCudaMemoryUsage("After Memory Free");
    printCpuMemoryUsage("After Memory Free");
}
