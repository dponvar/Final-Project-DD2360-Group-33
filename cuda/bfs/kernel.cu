/***********************************************************************************
 * Explanation:
 * This CUDA kernel is used for performing a parallel breadth-first search (BFS) on a graph. 
 * It processes each node in parallel, updating the cost (distance from the source node) for 
 * each of its unvisited neighbors. The kernel also uses flags (masks) to track which nodes 
 * are being explored and which need to be updated in the next iteration. It is designed to 
 * operate efficiently on the GPU using parallel threads and unified memory, allowing for 
 * faster traversal of the graph.
 ***********************************************************************************/


#ifndef _KERNEL_H_
#define _KERNEL_H_


__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool* g_graph_visited, int* g_cost, int no_of_nodes) 
{
    // Calculate the thread ID based on block and thread indices
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    // Check if the thread ID is within valid range and if the node is marked for exploration
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        
        // Mark the current node as no longer needing exploration
        g_graph_mask[tid] = false;

        // Iterate over all edges (neighbors) of the current node
        for (int i = g_graph_nodes[tid].starting; i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++) {
            
            // Get the ID of the neighboring node
            int id = g_graph_edges[i];

            // If the neighboring node has not been visited yet
            if (!g_graph_visited[id]) {
                
                // Update the cost for the neighboring node (distance from the source node)
                g_cost[id] = g_cost[tid] + 1;
                
                // Mark the neighboring node for updating in the next iteration
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
