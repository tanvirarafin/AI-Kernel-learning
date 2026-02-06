# Module 06: Graphs

Welcome to the final module in this introductory series. We now arrive at the **Graph**, the most flexible and powerful data structure. While trees are a specific type of graph, the world of general graphs is much larger.

## Introduction to Graphs

A graph is a data structure consisting of a set of **vertices** (or nodes) and a set of **edges** that establish connections between the vertices.

Unlike trees, graphs are not necessarily hierarchical. They can have:
-   **Cycles:** A path that starts and ends at the same vertex (e.g., A -> B -> C -> A).
-   **Disconnected Components:** A graph can be a collection of separate sub-graphs.
-   **Directed or Undirected Edges:** Edges can be one-way (directed) or two-way (undirected).

**Analogy:** A social network is a perfect graph. Each person is a vertex, and a "friendship" is an edge. A map of airline routes is another: cities are vertices, and flights are edges.

![Graph Diagram](https://i.imgur.com/qU3T7C4.png)

### Graph Representation

How we store a graph in memory is crucial. There are two main ways:

1.  **Adjacency Matrix:** A V x V matrix (where V is the number of vertices) where `matrix[i][j] = 1` if an edge exists between vertex `i` and `j`.
    -   **Pros:** Fast to check if an edge exists between two specific vertices (O(1)).
    -   **Cons:** Requires V² space, which is very inefficient for "sparse" graphs (graphs with few edges).

2.  **Adjacency List:** An array (or vector) of lists. The list at index `i` of the array stores all the vertices that are adjacent to vertex `i`.
    -   **Pros:** Space efficient, requiring O(V + E) space (where E is the number of edges). Most real-world graphs are sparse, making this the standard representation.
    -   **Cons:** Slower to check for a specific edge (O(k), where k is the number of neighbors).

**For this module, we will use an Adjacency List.**

---

## Graph Traversal: Breadth-First Search (BFS)

BFS is an algorithm for traversing or searching a graph. It explores the graph "layer by layer," meaning it visits all of a node's immediate neighbors before moving on to the next level of neighbors.

**Analogy:** Imagine dropping a stone in a pond. The ripples spread out in expanding circles. The ripple hits all points at 1 meter away, then all points at 2 meters away, and so on. This is how BFS explores a graph.

### How BFS Works (using a Queue)

The FIFO (First-In, First-Out) nature of a **Queue** is perfect for managing the "layers" of a BFS traversal.

1.  Choose a starting vertex. Add it to a **Queue** and mark it as visited.
2.  While the queue is not empty, do the following:
    a. Dequeue a vertex. Let's call it `current_vertex`.
    b. Process `current_vertex` (e.g., print its value).
    c. Look at all of the neighbors of `current_vertex`. For each neighbor that has **not yet been visited**:
        i. Mark it as visited.
        ii. Add it to the queue.

### Key Property of BFS

In an **unweighted graph**, BFS is guaranteed to find the **shortest path** (in terms of number of edges) from the starting vertex to all other reachable vertices.

---

## This Module's Task: Implement a Graph and BFS

For this module, you will implement a simple graph structure and a BFS traversal method.

**Task:**
1.  Create a `Graph` class that uses an adjacency list to store its structure.
2.  Implement an `addEdge` method to add connections between vertices.
3.  Implement a `BFS` method that takes a starting vertex and prints the nodes in the order they are visited.

**Goal:**
The goal is to build a simple graph and then run a BFS traversal to see how it explores the graph layer by layer, starting from a given node. Check the `main.cpp` file to see this in action.
