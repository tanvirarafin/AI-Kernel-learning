# Module 05: Trees

Welcome to Module 05. We are now moving from linear data structures (like Arrays and Linked Lists) to **hierarchical** data structures. The most common hierarchical structure is the **Tree**.

## Introduction to Trees

A tree is a collection of nodes connected by edges, representing a hierarchical relationship.

**Analogy:** Think of a family tree or a company's organizational chart. There's a single person at the top (the root), and the hierarchy flows down from there.

![Tree Diagram](https://i.imgur.com/3YQlA2E.png)

### Key Terminology

-   **Node:** An element in the tree that contains a value.
-   **Root:** The top-most node of the tree. A tree has only one root.
-   **Edge:** The link between a parent node and a child node.
-   **Parent:** A node that has one or more "child" nodes.
-   **Child:** A node that has a "parent" node.
-   **Leaf:** A node that has no children.
-   **Subtree:** A tree consisting of a node and all of its descendants.
-   **Height:** The length of the longest path from the root to a leaf node.

## Binary Trees & Binary Search Trees (BST)

A **Binary Tree** is a specific type of tree where every node has **at most two children**: a `left` child and a `right` child.

A **Binary Search Tree (BST)** is a special kind of Binary Tree that follows a very specific ordering property. This property is the key to its efficiency.

**The BST Property:**
For any given node `N` in the tree:
1.  All values in `N`'s **left subtree** are **less than** `N`'s value.
2.  All values in `N`'s **right subtree** are **greater than** `N`'s value.
3.  Both the left and right subtrees must also be binary search trees.

![BST Diagram](https://i.imgur.com/M6LplND.png)

The primary advantage of a BST is that operations like search, insertion, and deletion can be very fast. In a reasonably balanced BST, these operations have a time complexity of **O(log n)**, the same as binary search on a sorted array, but with the added flexibility of fast insertion and deletion.

## Tree Traversal

"Traversing" a tree means visiting every node in a specific order. There are three common ways to do this:

1.  **In-order Traversal:** `Left -> Root -> Right`
    -   Visit the left subtree, then the root node, then the right subtree.
    -   **For a BST, an in-order traversal visits the nodes in ascending, sorted order.** This is an extremely important property.

2.  **Pre-order Traversal:** `Root -> Left -> Right`
    -   Visit the root node, then the left subtree, then the right subtree.
    -   This is useful for creating a copy of a tree.

3.  **Post-order Traversal:** `Left -> Right -> Root`
    -   Visit the left subtree, then the right subtree, then the root node.
    -   This is useful for deleting a tree, as you delete the children before you delete the parent.

---

## This Module's Task: Implement a BST

For this module, you will implement a basic Binary Search Tree.

**Task:**
1.  Create a `TreeNode` structure.
2.  Implement an `insert` function that adds a new value to the tree while maintaining the BST property.
3.  Implement an `inOrderTraversal` function.

**Goal:**
The goal is to insert a series of numbers into the BST in an arbitrary order and then use the in-order traversal to print them out. If implemented correctly, the numbers will be printed in sorted order, demonstrating the power of the BST's structure.

Check the `main.cpp` file for the implementation.
