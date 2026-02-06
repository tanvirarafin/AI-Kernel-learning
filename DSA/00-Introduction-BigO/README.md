# Module 00: Introduction to DSA & Big O Notation

Welcome to the first module of your Data Structures and Algorithms journey! This module introduces the two foundational concepts that every programmer must understand to write efficient and scalable code.

## What are Data Structures?

A **Data Structure** is a systematic way of organizing and storing data in a computer so that it can be accessed and modified efficiently.

Think about a library. A library doesn't just throw all its books into a big pile. It organizes them using structures: by genre, by author's last name (alphabetically), or using a card catalog system. Each method of organization has its own benefits:
-   Organizing by genre is great if you want to browse for a new sci-fi book.
-   Organizing alphabetically is great if you know the exact title or author you're looking for.

Similarly, in computer science, we use different data structures like arrays, linked lists, and trees to organize data for different purposes. The choice of data structure can have a massive impact on how fast your program runs.

## What are Algorithms?

An **Algorithm** is a step-by-step set of instructions or rules to be followed to solve a specific problem or to perform a computation.

If the data structure is the "library organization," the algorithm is "how you find a specific book."
-   **Algorithm 1 (Inefficient):** Start at the first shelf, look at every single book in the library one by one until you find the one you want.
-   **Algorithm 2 (Efficient):** Look up the book in the computer catalog, find its exact shelf number, and go directly there.

Clearly, Algorithm 2 is much faster. Algorithms are all about finding the most efficient set of steps to solve a problem.

## Why Do They Matter? The Concept of Efficiency

Imagine you're building a social media app with a billion users. When a user logs in, you need to find their profile from a massive dataset. If you use an inefficient "one-by-one" search algorithm, that user might have to wait for minutes to log in. If you use an efficient algorithm, it could take milliseconds. This is the difference that DSA makes.

We measure this efficiency primarily in terms of:
1.  **Time Complexity:** How much longer does the algorithm take to run as the input data grows?
2.  **Space Complexity:** How much more memory does the algorithm need as the input data grows?

For now, we will focus on **Time Complexity**.

## Big O Notation: Measuring Time Complexity

**Big O Notation** is the language we use to describe the efficiency of an algorithm. It describes the *worst-case scenario* and tells us how the runtime of an algorithm grows as the input size (denoted by 'n') increases.

Here are the most common ones you'll see:

### 1. O(1) — Constant Time
The algorithm takes the same amount of time to run, regardless of the size of the input.
-   **Analogy:** Accessing the first element of an array. It doesn't matter if the array has 10 elements or 10 million; getting the first one takes the same amount of time.
-   **Example:** `int first = myArray[0];`

### 2. O(n) — Linear Time
The runtime of the algorithm grows in direct proportion to the size of the input 'n'.
-   **Analogy:** Reading a book. A 200-page book will take you twice as long to read as a 100-page book.
-   **Example:** Searching for an element in an array by checking every single element from start to finish.

```cpp
// O(n) example
for (int i = 0; i < n; i++) {
  if (myArray[i] == target) {
    return i; // Found it!
  }
}
```

### 3. O(n²) — Quadratic Time
The runtime of the algorithm is proportional to the square of the input size. This is common in algorithms that involve nested loops over the data.
-   **Analogy:** Shaking hands. If you have a group of 'n' people and everyone needs to shake hands with everyone else, you'll have roughly n² handshakes.
-   **Example:** Checking every possible pair of elements in an array.

```cpp
// O(n^2) example
for (int i = 0; i < n; i++) {
  for (int j = 0; j < n; j++) {
    // do something with the pair (myArray[i], myArray[j])
  }
}
```

**O(n²) algorithms are generally considered slow for large inputs and should be avoided if a more efficient alternative (like O(n) or O(log n)) exists.**

---

## This Module's Task

The `main.cpp` file for this module demonstrates the real-world difference between an **O(n²)** and an **O(n)** algorithm for the same problem: "Given a list of numbers, find if any pair of numbers sums up to a target value."

-   **The Naive Approach:** Uses two nested loops to check every single pair. This is **O(n²)**.
-   **The Efficient Approach:** Uses a clever trick with a hash set to solve the problem in a single pass. This is **O(n)**.

Study the code and the comments. Try compiling and running it yourself. You will see a dramatic performance difference as you increase the size of the input data, making the abstract concept of Big O very tangible.
