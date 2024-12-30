/***********************************************************************************
 * Explanation:
 * This CUDA kernel is used in conjunction with the breadth-first search (BFS) traversal 
 * to finalize the exploration of nodes that were updated in the previous kernel (Kernel). 
 * It checks if a node is marked for updating, sets it as visited, updates the exploration 
 * mask, and sets a flag indicating that at least one node was updated in the current iteration. 
 * This kernel is critical for controlling the flow of BFS and ensuring that the traversal continues 
 * until all reachable nodes have been processed.
 ***********************************************************************************/

#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void
Kernel2(bool* g_graph_mask, bool* g_updating_graph_mask, bool* g_graph_visited, bool* g_over, int no_of_nodes)
{
    // Calculate the thread ID based on block and thread indices
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    // Check if the thread ID is within valid range and if the node is marked for updating
    if (tid < no_of_nodes && g_updating_graph_mask[tid]) {

        // Mark the current node as explored
        g_graph_mask[tid] = true;
        g_graph_visited[tid] = true;

        // Set the 'over' flag to true indicating that the BFS should continue
        *g_over = true;

        // Reset the updating flag for the current node
        g_updating_graph_mask[tid] = false;
    }
}

#endif
