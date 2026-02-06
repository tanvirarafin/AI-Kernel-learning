# Module 01: Arrays & Strings

Welcome to Module 01. Here, we will take a deeper dive into the most fundamental data structure of all: the **Array**. In C++, you will most commonly work with arrays via the `std::vector` container, which provides a safer and more flexible alternative to built-in C-style arrays.

## The Array Data Structure

An array is a collection of items stored at **contiguous memory locations**. This means that the elements of an array are right next to each other in memory, like houses on a street.

![Array Memory Layout](https://i.imgur.com/n7T2a1s.png)

This structure leads to some very important performance characteristics.

### Array Operations and Time Complexity

-   **Access by Index: O(1) - Constant Time**
    -   If you want to get the element at index `i`, the computer can calculate its exact memory address instantly (`start_address + i * item_size`).
    -   This is why accessing `my_vector[5]` is incredibly fast, regardless of whether the vector has 10 elements or 10 million.

-   **Search for a Value: O(n) - Linear Time**
    -   If you want to find if a specific value exists in an unsorted array, you have no choice but to look at every single element, one by one, until you find it (or reach the end).
    -   In the worst case, you have to scan the entire array, so the time taken is proportional to the number of elements, `n`.

-   **Insertion / Deletion: O(n) - Linear Time**
    -   Because an array must be contiguous, inserting an element at the beginning or in the middle requires you to shift all subsequent elements one position to the right to make space.
    -   Similarly, deleting an element requires shifting all subsequent elements to the left to close the gap.
    -   In the worst case (inserting at the beginning), this requires `n` shifts. The only exception is adding/removing from the *end* of a dynamic array (like `std::vector`), which is typically `O(1)`.

## This Module's Task: Find the Missing Number

This task is a classic problem that leverages a clever mathematical trick to solve it in linear time, avoiding slower methods like sorting.

**Problem:** You are given an array containing `n` distinct numbers taken from the sequence `0, 1, 2, ..., n`. Because the array only has `n` numbers out of the `n+1` possible numbers in the sequence, exactly one number is missing. Your task is to find that missing number.

**Example 1:**
-   Input: `nums = [3, 0, 1]`
-   In this case, `n = 3` (since there are 3 numbers). The full sequence of numbers from 0 to n is `[0, 1, 2, 3]`.
-   Comparing the sequence to our input, the number `2` is missing.
-   Output: `2`

**Example 2:**
-   Input: `nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]`
-   `n = 9`. The full sequence is `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
-   The number `8` is missing.
-   Output: `8`

### The Efficient Solution: Gauss's Summation Formula

The 18th-century mathematician Carl Friedrich Gauss discovered that the sum of an unbroken sequence of integers from 1 to `n` can be calculated with the formula: `n * (n + 1) / 2`.

We can adapt this for our sequence from `0` to `n`.

1.  Calculate the **expected sum** if no numbers were missing. This would be the sum of all numbers from `0` to `n`.
2.  Calculate the **actual sum** of the numbers currently in the input array.
3.  The difference between the `expected sum` and the `actual sum` is precisely the number that is missing.

This approach is highly efficient because it only requires us to iterate through the array once to calculate the actual sum, making it an **O(n)** solution.

Now, examine the `main.cpp` file to see this solution implemented.
