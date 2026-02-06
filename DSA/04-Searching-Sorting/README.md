# Module 04: Searching & Sorting Algorithms

Welcome to Module 04. So far, we've focused on how to store data. Now, we'll focus on **algorithms**: common procedures for manipulating that data. Searching and sorting are two of the most fundamental operations in computer science.

---

## Searching Algorithms

### Binary Search: A "Divide and Conquer" Strategy

In Module 01, we saw that a standard search on an array takes O(n) time (Linear Search). If the array is **sorted**, we can do much, much better by using **Binary Search**.

Binary Search has a time complexity of **O(log n)**, which is incredibly fast. For a list of 1 billion items, a linear search might take 1 billion operations in the worst case, while a binary search will take only about 30!

**Requirement:** The data structure (e.g., an array) **must be sorted**.

**The Algorithm:**

1.  Start with the entire sorted array.
2.  Find the **middle** element of the current search range.
3.  Compare the middle element with the target value:
    -   If they are **equal**, you've found the element!
    -   If the middle element is **greater than** the target, you know the target must be in the **left half** of the array. Discard the right half.
    -   If the middle element is **less than** the target, you know the target must be in the **right half** of the array. Discard the left half.
4.  Repeat steps 2 and 3 on the new, smaller range until the element is found or the range becomes empty.

**Analogy:** Imagine guessing a number I'm thinking of from 1 to 100. Instead of guessing 1, 2, 3... (linear), you guess 50. I say "too high." You now know the number is between 1 and 49. You guess the middle, 25. I say "too low." You now know the number is between 26 and 49. You are cutting the problem in half with every guess.

---

## Sorting Algorithms

Sorting data is essential for enabling efficient searching and for presenting data in a human-readable order. There are many sorting algorithms, each with different trade-offs. We will focus on **Merge Sort**, a perfect example of the "Divide and Conquer" paradigm.

### Merge Sort: A Recursive Approach

Merge Sort is an efficient, stable, and widely used sorting algorithm with a time complexity of **O(n log n)**.

**The Algorithm:**

It's a recursive algorithm that breaks down into two main parts:

1.  **Divide (Recursive Step):**
    -   If the list has 0 or 1 elements, it is already considered sorted, so return.
    -   Otherwise, split the unsorted list into two halves of about equal size.
    -   Call Merge Sort on each of the two halves recursively.

2.  **Conquer (Merge Step):**
    -   Once the two halves are sorted (because the recursive calls have returned), you need to **merge** them back together into one single sorted list.
    -   **The Merge Process:**
        -   Create a new, empty list.
        -   Use two pointers, one for each of the sorted halves.
        -   Compare the elements at both pointers. Copy the smaller of the two elements into the new list and advance the pointer of the list you copied from.
        -   Repeat this until one of the lists is empty.
        -   Finally, copy the remaining elements from the non-empty list into the new list.
        -   The new list is now the merged, sorted result of the two halves.

![Merge Sort Diagram](https://i.imgur.com/gT1ikdF.png)

---

## This Module's Task

In the `main.cpp` file, you will find implementations for both **Binary Search** and **Merge Sort**. The code will:
1. Start with an unsorted vector.
2. Sort the vector using Merge Sort.
3. Use Binary Search to find a value in the now-sorted vector.

This demonstrates the powerful relationship between sorting and searching.
