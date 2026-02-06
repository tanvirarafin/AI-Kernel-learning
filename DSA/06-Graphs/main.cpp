#include <iostream>
#include <vector>
#include <queue>
#include <list> // Can use std::list or std::vector for the adjacency list

/**
 * TASK: Implement a Graph and Breadth-First Search (BFS)
 *
 * This file contains:
 * 1. A Graph class represented by an adjacency list.
 * 2. A method to add an edge to the graph.
 * 3. A BFS method to traverse the graph from a starting vertex.
 */

class Graph {
private:
    int num_vertices;
    // Adjacency List: An array of lists.
    // `adj[i]` contains a list of all vertices adjacent to vertex `i`.
    std::vector<std::list<int>> adj;

public:
    // Constructor to initialize the graph with a given number of vertices.
    Graph(int vertices) : num_vertices(vertices) {
        adj.resize(num_vertices);
    }

    /**
     * Adds an edge to an undirected graph.
     * We add the edge in both directions (from src to dest, and dest to src).
     */
    void addEdge(int src, int dest) {
        adj[src].push_back(dest);
        // Since this is an undirected graph, add the reverse edge as well.
        // For a directed graph, you would omit the next line.
        adj[dest].push_back(src);
    }

    /**
     * Performs Breadth-First Search traversal starting from a given vertex.
     *
     * COMPLEXITY:
     *   - Time: O(V + E) - where V is vertices and E is edges. We visit each
     *           vertex and edge once.
     *   - Space: O(V) - For the `visited` array and the `queue` in the worst case.
     *
     * @param start_vertex The vertex from which to start the traversal.
     */
    void BFS(int start_vertex) {
        // A boolean vector to keep track of visited vertices.
        // Initially, all vertices are marked as not visited.
        std::vector<bool> visited(num_vertices, false);

        // A queue for the BFS traversal.
        std::queue<int> q;

        // Start with the given `start_vertex`. Mark it as visited and enqueue it.
        visited[start_vertex] = true;
        q.push(start_vertex);

        std::cout << "BFS Traversal starting from vertex " << start_vertex << ": ";

        // Loop as long as the queue is not empty.
        while (!q.empty()) {
            // Dequeue a vertex from the front of the queue and print it.
            int current_vertex = q.front();
            q.pop();
            std::cout << current_vertex << " ";

            // Get all adjacent vertices of the dequeued vertex `current_vertex`.
            // For each adjacent vertex...
            for (int neighbor : adj[current_vertex]) {
                // ...if it has not been visited yet...
                if (!visited[neighbor]) {
                    // ...mark it as visited and enqueue it.
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "--- Module 06: Graphs and BFS ---" << std::endl << std::endl;

    // Create a graph with 7 vertices.
    Graph g(7);

    /*
      Let's create the following graph:
            0 --- 1
            | \   |
            |  \  |
            2 -- 3 --- 4
                 |
                 5 --- 6
    */

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(0, 3);
    g.addEdge(1, 3);
    g.addEdge(2, 3);
    g.addEdge(3, 4);
    g.addEdge(3, 5);
    g.addEdge(5, 6);

    // Perform BFS starting from vertex 0.
    // Expected Output (order of neighbors at a level can vary):
    // 0 1 2 3 4 5 6
    g.BFS(0);
    
    std::cout << std::endl;
    std::cout << "Notice how BFS explores layer by layer:" << std::endl;
    std::cout << "Layer 0: {0}" << std::endl;
    std::cout << "Layer 1 (neighbors of 0): {1, 2, 3}" << std::endl;
    std::cout << "Layer 2 (neighbors of 1, 2, 3): {4, 5}" << std::endl;
    std::cout << "Layer 3 (neighbors of 5): {6}" << std::endl;

    return 0;
}
