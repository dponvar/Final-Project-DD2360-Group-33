/*
Explanation:
This program implements a parallel Breadth-First Search (BFS) algorithm on a graph using CUDA for parallel computation on the GPU. 
It reads the graph data from an input file, applies BFS to traverse the graph, and outputs the shortest path (cost) to each node from the source node.

Key Concepts:
- Unified memory: It allows both the CPU and GPU to access the same memory region without explicit data transfer between the two.
- CUDA kernels: Functions executed on the GPU in parallel to accelerate computations.

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512  // Maximum number of threads allowed per block in the CUDA kernel. 

int no_of_nodes;  // Variable to store the number of nodes in the graph.
int edge_list_size;  // Variable to store the size of the edge list.
FILE *fp;  // File pointer to handle the input graph data file.

struct Node {
    int starting;  // Starting node of the edges for this node.
    int no_of_edges;  // Number of edges this node has.
};

#include "kernel.cu"  // Includes the CUDA kernel for BFS traversal.
#include "kernel2.cu"  // Includes the second CUDA kernel to update the graph during traversal.

void BFSGraph(int argc, char** argv);  // Function prototype for BFS execution on the graph.

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    no_of_nodes = 0;
    edge_list_size = 0;
    printf("Initializing BFSGraph...\n");
    BFSGraph(argc, argv);  // Call BFSGraph function to start processing the graph.
    printf("Execution completed.\n");
}

void Usage(int argc, char** argv) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);  // Print usage information if the program is not called correctly.
}

////////////////////////////////////////////////////////////////////////////////
// BFSGraph: Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) {
    char *input_f;
    if (argc != 2) {  // Ensure the correct number of arguments are provided.
        Usage(argc, argv);  // Print usage if incorrect arguments are provided.
        exit(0);
    }
    
    input_f = argv[1];  // Get the input file name (graph data).
    printf("Reading File: %s\n", input_f);
    fp = fopen(input_f, "r");  // Open the file to read the graph data.
    if (!fp) {  // Check if the file was successfully opened.
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;  // Initialize the source node for BFS. By default, it is set to node 0.
    if (fscanf(fp, "%d", &no_of_nodes) != 1) {  // Read the number of nodes in the graph from the file.
        fprintf(stderr, "Error reading number of nodes.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    printf("Number of nodes: %d\n", no_of_nodes);

    // Calculate the number of blocks and threads per block for CUDA kernel execution.
    int num_of_blocks = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;  // Ensure enough blocks to cover all nodes.
    int num_of_threads_per_block = (no_of_nodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : no_of_nodes;
    printf("Execution Configuration -> Blocks: %d, Threads per block: %d\n", num_of_blocks, num_of_threads_per_block);

    // Allocate unified memory for graph structures.
    Node* d_graph_nodes;  // Array of nodes in the graph.
    bool *d_graph_mask, *d_updating_graph_mask, *d_graph_visited, *d_over;  // Arrays for BFS control (masks and visited flags).
    int* d_graph_edges, *d_cost;  // Arrays for storing edge data and the cost (distance) from source to each node.

    // Allocating unified memory ensures the GPU and CPU can both access the memory without explicit transfers.
    if (cudaMallocManaged(&d_graph_nodes, sizeof(Node) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_graph_mask, sizeof(bool) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_updating_graph_mask, sizeof(bool) * no_of_nodes) != cudaSuccess ||
        cudaMallocManaged(&d_graph_visited, sizeof(bool) * no_of_nodes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory for graph structures.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Initialize the graph nodes with data read from the file (node connections).
    int start, edgeno;   
    for (unsigned int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {  // Read the starting node and the number of edges for each node.
            fprintf(stderr, "Error reading node information.\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        d_graph_nodes[i].starting = start;
        d_graph_nodes[i].no_of_edges = edgeno;
        d_graph_mask[i] = false;  // Initially, no node is marked for BFS.
        d_updating_graph_mask[i] = false;  // No node is being updated initially.
        d_graph_visited[i] = false;  // No node is visited at the beginning.
    }
    printf("Graph nodes initialized.\n");

    // Read the source node from the file (this is the starting point for BFS).
    fscanf(fp, "%d", &source);
    source = 0;  // Set the source node to 0 for this example.
    printf("Source node: %d\n", source);

    // Mark the source node as visited and add it to the graph mask to start BFS from it.
    d_graph_mask[source] = true;
    d_graph_visited[source] = true;

    // Read the size of the edge list from the file.
    if (fscanf(fp, "%d", &edge_list_size) != 1) {
        fprintf(stderr, "Error reading edge list size.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    printf("Edge list size: %d\n", edge_list_size);

    // Allocate unified memory for the graph's edges and read the edge data.
    cudaMallocManaged(&d_graph_edges, sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        int id, cost;
        if (fscanf(fp, "%d %d", &id, &cost) != 2) {  // Read edge information (id and cost).
            fprintf(stderr, "Error reading edge information.\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        d_graph_edges[i] = id;  // Store the edge IDs in the array.
    }
    printf("Graph edges initialized.\n");

    if (fp) fclose(fp);  // Close the file after reading.
    printf("File reading completed.\n");

    // Allocate memory for the cost array and a flag to check when BFS is finished.
    cudaMallocManaged(&d_cost, sizeof(int) * no_of_nodes);
    cudaMallocManaged(&d_over, sizeof(bool));
    for (int i = 0; i < no_of_nodes; i++) d_cost[i] = -1;  // Initialize the cost of all nodes to -1 (not visited).
    d_cost[source] = 0;  // Set the cost of the source node to 0 (starting point).
    printf("Memory allocated and copied to GPU.\n");

    // Set up the grid and block dimensions for CUDA kernel execution.
    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    int k = 0;
    printf("Starting BFS traversal.\n");
    bool *stop;
    cudaMallocManaged(&stop, sizeof(bool));

    // Perform BFS traversal in a loop until no more updates are made.
    do {
        *stop = false;  // Set the stop flag to false before each iteration.
        printf("Kernel iteration %d\n", k);
        
        // Call the CUDA kernels to process the graph. These kernels perform the BFS traversal in parallel.
        Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, 
                                  d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
        
        Kernel2<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask, 
                                   d_graph_visited, stop, no_of_nodes);
        
        cudaDeviceSynchronize();  // Synchronize the GPU to ensure all tasks are completed before the next iteration.
        
        k++;  // Increment the iteration counter.
    } while (*stop);  // Continue looping until no more nodes are being updated.

    printf("Kernel executed %d times.\n", k);

    // Write the final result to a file.
    FILE *fpo = fopen("result_managed.txt", "w");
    printf("%d", source);
    int sourceCost = d_cost[source];  // Get the cost to reach the source node.
    printf("Source node: %d, Cost: %d\n", source, sourceCost);
    for (int i = 0; i < no_of_nodes; i++) fprintf(fpo, "%d) cost:%d\n", i, d_cost[i]);  // Write the cost for all nodes to a file.
    fclose(fpo);
    printf("Results stored in newresult.txt.\n");

    // Free all allocated GPU memory after the BFS traversal is complete.
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
    cudaFree(d_over);
    cudaFree(stop);
}
