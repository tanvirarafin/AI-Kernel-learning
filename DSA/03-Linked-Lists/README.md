# Module 03: Linked Lists

In this module, we'll explore the **Linked List**, a fundamental data structure that offers a different set of trade-offs compared to arrays.

## The Linked List Data Structure

A linked list is a linear collection of data elements whose order is not given by their physical placement in memory. Instead, each element, called a **Node**, points to the next one.

A **Node** consists of two parts:
1.  **Data:** The value stored in the node (e.g., an integer, a string).
2.  **Next Pointer:** A reference (or pointer) to the next node in the sequence. The `next` pointer of the last node in the list points to `null` (or `nullptr` in C++), indicating the end of the list.

The entry point to the list is a pointer to the first node, which is called the **Head**.

**Analogy:** Think of a scavenger hunt. Each clue (`Node`) contains a piece of information (`data`) and tells you where to find the next clue (`next` pointer). The `head` is the first clue you are given.

![Linked List Diagram](https://i.imgur.com/L7s4z9Y.png)

### Linked Lists vs. Arrays

Understanding the trade-offs between linked lists and arrays is crucial.

| Feature             | Array (`std::vector`)                                  | Linked List                                             |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| **Memory Layout**   | **Contiguous** (elements are side-by-side)             | **Non-contiguous** (nodes can be scattered in memory)   |
| **Access (by Index)** | **O(1)** - Instant access via index (e.g., `vec[5]`).   | **O(n)** - Must traverse from the head to find the Nth element. |
| **Search (by Value)** | **O(n)** - Must scan the elements one-by-one.           | **O(n)** - Must scan the nodes one-by-one.             |
| **Insertion/Deletion** | **O(n)** - Slow. May require shifting all elements.    | **O(1)** - Fast, **if** you have a pointer to the node. Just redirect pointers. |
| **Size**            | Fixed or dynamically resizing (can be slow).           | Dynamic. Grows and shrinks one node at a time easily. |

In summary, use an Array/Vector when you need fast random access. Use a Linked List when you need fast insertions and deletions and don't require frequent random access.

---

## This Module's Task: Reverse a Singly Linked List

This is one of the most common interview questions related to linked lists. It tests your ability to manipulate pointers and manage state without losing track of nodes.

**Problem:** Given the `head` of a singly linked list, reverse the list and return the new head.

**Example:**
-   **Input:** A list that looks like `1 -> 2 -> 3 -> 4 -> 5 -> NULL`
-   **Output:** The new head should point to a list that looks like `5 -> 4 -> 3 -> 2 -> 1 -> NULL`

### The Iterative Solution (Pointer Dance)

To reverse the list, we need to change the direction of the `next` pointers for every node. We can do this by iterating through the list and using three pointers to keep track of our state:

1.  `previous`: This will track the node that came *before* the current one. It will eventually become the `next` node for our `current` node. Starts as `nullptr`.
2.  `current`: The node we are currently visiting. Starts at the `head` of the list.
3.  `next_temp`: This is a temporary pointer to store the *original* next node before we overwrite `current->next`.

**The Algorithm:**

Initialize `previous = nullptr` and `current = head`. Then, loop as long as `current` is not `nullptr`:

1.  **Store the next node:** `next_temp = current->next;` (We need this so we don't lose the rest of the list).
2.  **Reverse the pointer:** `current->next = previous;` (This is the key step! The current node now points "backwards").
3.  **Advance the pointers:** Move `previous` and `current` one step forward for the next iteration.
    -   `previous = current;`
    -   `current = next_temp;`

When the loop finishes, `current` will be `nullptr`, and `previous` will be pointing at the last node of the original list, which is the new `head` of the reversed list.

Study the `main.cpp` file to see how these pointers "dance" together to reverse the list.
