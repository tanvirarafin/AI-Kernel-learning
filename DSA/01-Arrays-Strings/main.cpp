#include <iostream>
#include <vector>
#include <numeric> // Required for std::accumulate

/**
 * TASK: Find the Missing Number
 *
 * Given an array `nums` containing `n` distinct numbers taken from the
 * sequence 0, 1, 2, ..., n, find the one number that is missing from the array.
 *
 * This implementation uses the efficient Gaussian Summation method.
 */

// =======================================================================================
// SOLUTION
//
// COMPLEXITY:
//   - Time: O(n) - Linear Time, because we iterate through the list once to get its sum.
//   - Space: O(1) - Constant Space, because we only use a few variables to store sums,
//           regardless of the input size.
//
// HOW IT WORKS:
// 1. The size of the input vector, `n`, also happens to be the largest number
//    in the complete sequence `0, 1, ..., n`.
// 2. We calculate the `expected_sum` of the complete sequence using the formula:
//    n * (n + 1) / 2.
// 3. We calculate the `actual_sum` of the numbers that are actually in the input vector.
//    `std::accumulate` is a convenient way to do this.
// 4. The difference, `expected_sum - actual_sum`, is the missing number.
// =======================================================================================
int findMissingNumber(const std::vector<int>& nums) {
    // The number of elements in the vector. This value is also the 'n' in the
    // sequence 0, 1, ..., n.
    int n = nums.size();

    // Calculate the expected sum of the full sequence from 0 to n.
    int expected_sum = n * (n + 1) / 2;

    // Calculate the actual sum of the elements in the input vector.
    // std::accumulate(start, end, initial_value)
    int actual_sum = std::accumulate(nums.begin(), nums.end(), 0);

    // The difference is the number that was left out.
    return expected_sum - actual_sum;
}

// Helper function to run a test case and print the result
void runTest(const std::vector<int>& test_vector) {
    int missing_number = findMissingNumber(test_vector);
    std::cout << "Input: { ";
    for (size_t i = 0; i < test_vector.size(); ++i) {
        std::cout << test_vector[i] << (i == test_vector.size() - 1 ? " " : ", ");
    }
    std::cout << "}" << std::endl;
    std::cout << "Missing number: " << missing_number << std::endl << std::endl;
}

int main() {
    std::cout << "--- Module 01: Find the Missing Number ---" << std::endl << std::endl;

    // Test Case 1
    std::vector<int> test1 = {3, 0, 1};
    runTest(test1); // Expected output: 2

    // Test Case 2
    std::vector<int> test2 = {9, 6, 4, 2, 3, 5, 7, 0, 1};
    runTest(test2); // Expected output: 8

    // Test Case 3: A simple case where 0 is missing
    std::vector<int> test3 = {1};
    runTest(test3); // Expected output: 0
    
    // Test Case 4: A simple case where 1 is missing
    std::vector<int> test4 = {0};
    runTest(test4); // Expected output: 1

    // Test Case 5: A larger, unordered list
    std::vector<int> test5 = {10, 2, 8, 4, 0, 1, 7, 6, 9, 3};
    runTest(test5); // Expected output: 5

    return 0;
}
