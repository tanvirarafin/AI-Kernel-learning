#include <iostream>
#include <string>
#include <stack>

/**
 * TASK: Balanced Parentheses
 *
 * Given a string `s` containing just the characters '(', ')', '{', '}', '[' and ']',
 * determine if the input string is valid.
 *
 * A string is valid if:
 * 1. Open brackets are closed by the same type of brackets.
 * 2. Open brackets are closed in the correct order.
 *
 * This implementation uses a std::stack.
 */

// =======================================================================================
// SOLUTION
//
// COMPLEXITY:
//   - Time: O(n) - Linear Time, because we iterate through the string once.
//   - Space: O(n) - Linear Space, because in the worst case (e.g., "(((((")
//           the stack could hold all the characters of the string.
//
// HOW IT WORKS:
// The logic is described in detail in the README.md file. We use the LIFO
// (Last-In, First-Out) property of the stack to ensure the most recently
// opened bracket is the first to be closed.
// =======================================================================================
bool isValid(const std::string& s) {
    // A stack to keep track of opening brackets.
    std::stack<char> bracket_stack;

    // Iterate over each character in the string.
    for (char c : s) {
        // If it's an opening bracket, push it onto the stack.
        if (c == '(' || c == '{' || c == '[') {
            bracket_stack.push(c);
        }
        // If it's a closing bracket...
        else if (c == ')' || c == '}' || c == ']') {
            // ...but the stack is empty, it means there's no matching opener. Invalid.
            if (bracket_stack.empty()) {
                return false;
            }

            // Get the last opened bracket from the top of the stack.
            char top = bracket_stack.top();

            // Check if the current closing bracket matches the last opening bracket.
            if ((c == ')' && top == '(') ||
                (c == '}' && top == '{') ||
                (c == ']' && top == '[')) {
                // It's a match! Pop the opening bracket off the stack and continue.
                bracket_stack.pop();
            } else {
                // It's a mismatch (e.g., "([)]"). Invalid.
                return false;
            }
        }
    }

    // After iterating through the whole string, the stack must be empty for the
    // string to be valid. If it's not empty, it means there are unclosed openers.
    return bracket_stack.empty();
}

// Helper function to run a test case and print the result
void runTest(const std::string& test_string) {
    bool result = isValid(test_string);
    std::cout << "Input: \"" << test_string << "\"" << std::endl;
    std::cout << "Is Valid: " << (result ? "true" : "false") << std::endl << std::endl;
}

int main() {
    std::cout << "--- Module 02: Balanced Parentheses ---" << std::endl << std::endl;

    // Test Case 1: Simple valid case
    runTest("()"); // Expected: true

    // Test Case 2: Multiple valid pairs
    runTest("()[]{}"); // Expected: true

    // Test Case 3: Nested valid pairs
    runTest("{[]}"); // Expected: true

    // Test Case 4: Simple invalid case
    runTest("(]"); // Expected: false

    // Test Case 5: Mismatched nested pairs
    runTest("([)]"); // Expected: false

    // Test Case 6: Unclosed opening bracket
    runTest("{["); // Expected: false
    
    // Test Case 7: Closing bracket with no opener
    runTest("}"); // Expected: false
    
    // Test Case 8: Long valid case
    runTest("({[]})"); // Expected: true

    return 0;
}
