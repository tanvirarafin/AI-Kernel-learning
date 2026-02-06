#include <iostream>
#include <vector>
#include <algorithm>

/**
 * TASK: Reverse a Singly Linked List
 *
 * This file implements the reversal of a singly linked list.
 * It includes the definition for a list node, a function to reverse the list,
 * and helper functions for creating and printing lists to test the solution.
 */

// Definition for a singly-linked list node.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// =======================================================================================
// SOLUTION
//
// COMPLEXITY:
//   - Time: O(n) - Linear Time, as we visit each node exactly once.
//   - Space: O(1) - Constant Space, as we only use a few pointers, not
//           proportional to the list size.
//
// HOW IT WORKS:
// We use three pointers: `previous`, `current`, and `next_temp`.
// The `current` pointer traverses the list, while `previous` tracks the node
// before it. In each step, we save the original next node, reverse the
// current node's pointer to point to `previous`, and then advance both
// `previous` and `current` pointers forward.
// =======================================================================================
ListNode* reverseList(ListNode* head) {
    ListNode* previous = nullptr;
    ListNode* current = head;

    while (current != nullptr) {
        // 1. Store the original next node so we don't lose the rest of the list.
        ListNode* next_temp = current->next;

        // 2. Reverse the current node's pointer to point to the previous node.
        current->next = previous;

        // 3. Move the `previous` and `current` pointers one step forward.
        previous = current;
        current = next_temp;
    }

    // When the loop ends, `current` is nullptr, and `previous` is the new head.
    return previous;
}

// =======================================================================================
// HELPER FUNCTIONS FOR TESTING
// =======================================================================================

// Helper function to create a linked list from a vector of integers.
ListNode* createList(const std::vector<int>& values) {
    if (values.empty()) {
        return nullptr;
    }
    ListNode* head = new ListNode(values[0]);
    ListNode* current = head;
    for (size_t i = 1; i < values.size(); ++i) {
        current->next = new ListNode(values[i]);
        current = current->next;
    }
    return head;
}

// Helper function to print a linked list.
void printList(ListNode* head) {
    ListNode* current = head;
    while (current != nullptr) {
        std::cout << current->val << " -> ";
        current = current->next;
    }
    std::cout << "NULL" << std::endl;
}

int main() {
    std::cout << "--- Module 03: Reverse a Linked List ---" << std::endl << std::endl;

    // --- SETUP ---
    // Create an initial list: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
    std::vector<int> initial_values = {1, 2, 3, 4, 5};
    ListNode* head = createList(initial_values);

    std::cout << "Original list:" << std::endl;
    printList(head);
    std::cout << std::endl;

    // --- REVERSE ---
    head = reverseList(head);

    // --- VERIFY ---
    // The list should now be: 5 -> 4 -> 3 -> 2 -> 1 -> NULL
    std::cout << "Reversed list:" << std::endl;
    printList(head);

    // It's important to free the allocated memory in a real application,
    // but we'll omit that here to keep the focus on the algorithm.

    return 0;
}
