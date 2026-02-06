#include <iostream>
#include <vector>
#include <algorithm> // For std::copy

// Forward declarations for the recursive merge sort
void mergeSort(std::vector<int>& arr);
void merge(std::vector<int>& arr, std::vector<int>& left, std::vector<int>& right);

/**
 * TASK 1: MERGE SORT
 * Sorts a vector of integers using the Merge Sort algorithm.
 *
 * COMPLEXITY:
 *   - Time: O(n log n) - It consistently divides the list and merging takes linear time.
 *   - Space: O(n) - Due to the temporary arrays used during the merging process.
 */
void mergeSort(std::vector<int>& arr) {
    // Base case: An array of size 0 or 1 is already sorted.
    if (arr.size() <= 1) {
        return;
    }

    // 1. DIVIDE
    // Find the middle of the array.
    int mid = arr.size() / 2;
    // Create two sub-arrays: left and right.
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());

    // 2. CONQUER (Recursive calls)
    // Recursively sort the left and right halves.
    mergeSort(left);
    mergeSort(right);

    // 3. MERGE
    // Merge the now-sorted left and right halves back into the original array.
    merge(arr, left, right);
}

// Helper function to merge two sorted arrays `left` and `right` into `arr`.
void merge(std::vector<int>& arr, std::vector<int>& left, std::vector<int>& right) {
    // Pointers for the left array, right array, and the main array `arr`.
    size_t i = 0, j = 0, k = 0;

    // Compare elements from left and right, and copy the smaller one to `arr`.
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    // Copy any remaining elements from the left array (if any).
    while (i < left.size()) {
        arr[k++] = left[i++];
    }

    // Copy any remaining elements from the right array (if any).
    while (j < right.size()) {
        arr[k++] = right[j++];
    }
}

/**
 * TASK 2: BINARY SEARCH
 * Searches for a `target` value in a SORTED vector. Returns the index if found,
 * otherwise returns -1.
 *
 * COMPLEXITY:
 *   - Time: O(log n) - Because we divide the search space in half with each step.
 *   - Space: O(1) - We only use a few variables for the pointers.
 */
int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        // Find the middle index to avoid potential overflow: left + (right - left) / 2
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid; // Target found
        } else if (arr[mid] < target) {
            left = mid + 1; // Target must be in the right half
        } else {
            right = mid - 1; // Target must be in the left half
        }
    }

    return -1; // Target not found
}


// Helper function to print a vector
void printVector(const std::string& title, const std::vector<int>& arr) {
    std::cout << title;
    for (int val : arr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "--- Module 04: Searching & Sorting ---" << std::endl << std::endl;

    // --- SETUP ---
    std::vector<int> my_vector = {38, 27, 43, 3, 9, 82, 10};
    printVector("Original vector: ", my_vector);
    std::cout << std::endl;

    // --- SORT ---
    std::cout << "1. Sorting the vector using Merge Sort..." << std::endl;
    mergeSort(my_vector);
    printVector("Sorted vector:   ", my_vector);
    std::cout << std::endl;

    // --- SEARCH ---
    std::cout << "2. Searching for a value using Binary Search..." << std::endl;
    int target = 43;
    int index = binarySearch(my_vector, target);

    if (index != -1) {
        std::cout << "Found target " << target << " at index " << index << "." << std::endl;
    } else {
        std::cout << "Target " << target << " not found in the vector." << std::endl;
    }

    // Search for a value that doesn't exist
    target = 5;
    index = binarySearch(my_vector, target);
     if (index != -1) {
        std::cout << "Found target " << target << " at index " << index << "." << std::endl;
    } else {
        std::cout << "Target " << target << " not found in the vector." << std::endl;
    }

    return 0;
}
