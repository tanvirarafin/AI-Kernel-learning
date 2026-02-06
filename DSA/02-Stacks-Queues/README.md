# Module 02: Stacks & Queues

In this module, we move on to two simple yet powerful linear data structures: **Stacks** and **Queues**. They are considered "Abstract Data Types" because their definition is more about the rules of how they behave rather than the underlying structure (they can be implemented using either arrays or linked lists).

---

## The Stack Data Structure

A Stack follows the **LIFO** principle: **Last-In, First-Out**.

**Analogy:** Think of a stack of plates. You can only add a new plate to the top, and you can only take a plate from the top. The last plate you put on the stack will be the first one you take off.

![Stack Analogy](https://i.imgur.com/uG5p2dJ.png)

### Core Stack Operations

-   **Push:** Add an element to the top of the stack.
-   **Pop:** Remove the element from the top of the stack.
-   **Top (or Peek):** Look at the element at the top of the stack without removing it.
-   **isEmpty:** Check if the stack is empty.

Stacks are incredibly useful for problems that involve "back-tracking" or "undoing" steps, such as parsing, navigating file system paths (`cd ..`), or managing function calls in a program (the "call stack").

---

## The Queue Data Structure

A Queue follows the **FIFO** principle: **First-In, First-Out**.

**Analogy:** Think of a line at a grocery store checkout. The first person to get in the line is the first person to be served.

![Queue Analogy](https://i.imgur.com/7nKkGkH.png)

### Core Queue Operations

-   **Enqueue:** Add an element to the *back* (end) of the queue.
-   **Dequeue:** Remove the element from the *front* of the queue.
-   **Front (or Peek):** Look at the element at the front of the queue without removing it.
-   **isEmpty:** Check if the queue is empty.

Queues are perfect for managing tasks in the order they were received, like a print queue, processing requests on a web server, or the Breadth-First Search algorithm for graphs, which we will see in a later module.

---

## This Module's Task: Balanced Parentheses

This is a classic computer science problem that is a perfect illustration of the power of stacks.

**Problem:** Given a string `s` containing just the characters `(`, `)`, `{`, `}`, `[` and `]`, determine if the input string is valid.

An input string is valid if:
1.  Open brackets must be closed by the same type of brackets. (e.g., `(` must be closed by `)`).
2.  Open brackets must be closed in the correct order. (e.g., `( [ ] )` is correct, but `( [ ) ]` is not).

**Example 1:**
-   Input: `"()[]{}"`
-   Output: `true`

**Example 2:**
-   Input: `"(]"`
-   Output: `false`

**Example 3:**
-   Input: `"{[]}"`
-   Output: `true`

### The Solution using a Stack

The LIFO nature of a stack is exactly what we need to solve this. The most recently opened bracket must be the first one to be closed.

1.  Create an empty stack.
2.  Iterate through the input string one character at a time.
3.  If the character is an **opening bracket** (`(`, `{`, or `[`), **push** it onto the stack.
4.  If the character is a **closing bracket** (`)`, `}`, or `]`):
    -   First, check if the stack is empty. If it is, you have a closing bracket with no opener, so the string is invalid.
    -   If the stack is not empty, look at the bracket at the `top` of the stack.
    -   If the closing bracket matches the opening bracket at the top (e.g., character is `)` and top is `(`), then everything is good. **Pop** the opening bracket off the stack and continue.
    -   If it does *not* match, the brackets are not balanced correctly. The string is invalid.
5.  After you have iterated through the entire string, check if the stack is empty.
    -   If the stack is **empty**, it means every opening bracket was successfully matched with a closing bracket. The string is **valid**.
    -   If the stack is **not empty**, it means there are leftover opening brackets that were never closed. The string is **invalid**.

Now, review the `main.cpp` file to see this logic in action.
