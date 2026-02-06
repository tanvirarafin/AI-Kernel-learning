#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <random>
#include <algorithm>

/**
 * PROBLEM: Given a vector of integers `data` and a target integer `sum`,
 * determine if there exists a pair of numbers in the vector that adds up to the sum.
 */

// =======================================================================================
// SOLUTION 1: The Naive, Brute-Force Approach
//
// COMPLEXITY: O(n^2) - Quadratic Time
//
// HOW IT WORKS:
// We use two nested loops. The outer loop picks an element, and the inner loop
// iterates through the rest of the elements to see if any pair sums up to the target.
//
// WHY IT'S O(n^2):
// For an input vector of size 'n', the outer loop runs 'n' times. For each of those
// iterations, the inner loop also runs roughly 'n' times. This results in n * n,
// or n^2, operations. This approach becomes extremely slow as 'n' gets large.
// =======================================================================================
bool hasPairWithSum_On2(const std::vector<int>& data, int sum) {
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = i + 1; j < data.size(); ++j) {
            if (data[i] + data[j] == sum) {
                return true;
            }
        }
    }
    return false;
}

// =======================================================================================
// SOLUTION 2: The Efficient, Optimized Approach
//
// COMPLEXITY: O(n) - Linear Time
//
// HOW IT WORKS:
// We iterate through the vector just ONCE. For each number `num`, we calculate its
// "complement" (the other number that would make the sum), which is `sum - num`.
// We store every number we've seen so far in a hash set (std::unordered_set).
//
// A hash set has an amazing property: adding and searching for an element is,
// on average, an O(1) constant time operation.
//
// So, for each `num`, we check the hash set to see if its complement already exists.
// - If it does, we've found our pair!
// - If it doesn't, we add the current `num` to the set and move to the next element.
//
// WHY IT'S O(n):
// We only pass through the list once. Each operation inside the loop (calculating
// the complement, searching the set, and inserting into the set) is O(1).
// Therefore, the total time complexity is proportional to the size of the input, 'n'.
// =======================================================================================
bool hasPairWithSum_On(const std::vector<int>& data, int sum) {
    std::unordered_set<int> seen_numbers;
    for (int num : data) {
        int complement = sum - num;
        // The .count() or .find() operation in a hash set is O(1) on average.
        if (seen_numbers.count(complement)) {
            return true;
        }
        // The .insert() operation is also O(1) on average.
        seen_numbers.insert(num);
    }
    return false;
}

// Helper function to run and time a function
template<typename Func>
void run_and_time(const std::string& name, Func func, const std::vector<int>& data, int sum) {
    auto start = std::chrono::high_resolution_clock::now();
    bool result = func(data, sum);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "Pair found: " << (result ? "Yes" : "No") << std::endl;
    std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    // --- SETUP ---
    // Generate a large vector of random numbers to test with.
    // Using a large number helps to make the performance difference obvious.
    const int DATA_SIZE = 20000;
    std::vector<int> data_to_test;
    data_to_test.reserve(DATA_SIZE);

    // Use a fixed seed for reproducibility
    std::mt19937 rng(0); 
    std::uniform_int_distribution<int> dist(0, 100000);

    for (int i = 0; i < DATA_SIZE; ++i) {
        data_to_test.push_back(dist(rng));
    }

    // A target sum that is unlikely to be found, forcing both algorithms
    // to do the maximum amount of work (the worst-case scenario).
    int target_sum = -1; 

    std::cout << "Testing with a dataset of " << DATA_SIZE << " numbers." << std::endl;
    std::cout << "The goal is to find if any pair sums to " << target_sum << "." << std::endl << std::endl;

    // --- EXECUTION ---
    // Run the efficient O(n) version
    run_and_time("O(n) - Efficient Approach", hasPairWithSum_On, data_to_test, target_sum);

    // Run the inefficient O(n^2) version
    run_and_time("O(n^2) - Naive Approach", hasPairWithSum_On2, data_to_test, target_sum);

    std::cout << "Notice the massive difference in execution time!" << std::endl;
    std::cout << "The O(n) algorithm is thousands of times faster for this input size." << std::endl;

    return 0;
}
