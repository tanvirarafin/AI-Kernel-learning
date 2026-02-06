#include <iostream>

/**
 * TASK: Implement a Binary Search Tree (BST)
 *
 * This file contains:
 * 1. The definition for a TreeNode.
 * 2. A function to insert a new value into the BST.
 * 3. A function to perform an in-order traversal of the BST.
 * 4. A main function to demonstrate that in-order traversal of a BST
 *    results in a sorted sequence.
 */

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

/**
 * Inserts a value into the Binary Search Tree.
 *
 * COMPLEXITY:
 *   - Time: O(log n) on average for a balanced tree, O(n) in the worst case (a skewed tree).
 *   - Space: O(log n) for the recursion call stack, O(n) in the worst case.
 *
 * @param root A pointer to the root of the tree (or subtree).
 * @param val The value to be inserted.
 * @return The root of the modified tree.
 */
TreeNode* insert(TreeNode* root, int val) {
    // If the tree/subtree is empty, create a new node and return it as the new root.
    if (root == nullptr) {
        return new TreeNode(val);
    }

    // If the value is less than the root's value, it belongs in the left subtree.
    if (val < root->val) {
        root->left = insert(root->left, val);
    }
    // If the value is greater than the root's value, it belongs in the right subtree.
    else { // We can ignore the case where val == root->val for this simple example
        root->right = insert(root->right, val);
    }

    // Return the (possibly unchanged) root pointer.
    return root;
}

/**
 * Performs an in-order traversal of the BST (Left -> Root -> Right).
 * This will print the node values in ascending sorted order.
 *
 * @param root The root of the tree to be traversed.
 */
void inOrderTraversal(TreeNode* root) {
    // Base case: If the node is null, do nothing and return.
    if (root == nullptr) {
        return;
    }

    // 1. Visit the left subtree.
    inOrderTraversal(root->left);

    // 2. Visit the root node (in this case, print its value).
    std::cout << root->val << " ";

    // 3. Visit the right subtree.
    inOrderTraversal(root->right);
}

int main() {
    std::cout << "--- Module 05: Binary Search Trees ---" << std::endl << std::endl;

    // Start with an empty tree (a null pointer).
    TreeNode* root = nullptr;

    std::cout << "Inserting the values: 50, 30, 20, 40, 70, 60, 80" << std::endl << std::endl;

    // Insert nodes into the BST.
    root = insert(root, 50);
    root = insert(root, 30);
    root = insert(root, 20);
    root = insert(root, 40);
    root = insert(root, 70);
    root = insert(root, 60);
    root = insert(root, 80);

    /*
     The resulting tree will look like this:
            50
           /  \
          30   70
         / \   / \
        20 40 60 80
    */

    std::cout << "Performing in-order traversal (Left -> Root -> Right):" << std::endl;

    // Perform an in-order traversal. The output should be sorted!
    inOrderTraversal(root);
    std::cout << std::endl << std::endl;

    std::cout << "Notice that even though the numbers were inserted out of order,"
    std::cout << "the in-order traversal prints them in a sorted sequence."
    std::cout << std::endl;

    // In a real application, you would need to deallocate the memory for the nodes.
    // This is often done with a post-order traversal.

    return 0;
}
