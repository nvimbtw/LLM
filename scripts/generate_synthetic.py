#!/usr/bin/env python3
"""
Generate synthetic thinking-trace training data.

Each file contains a series of <|thinking|>...<|/thinking|> blocks interleaved
with <|par|>...<|/par|> response content, wrapped in project markup tokens.

Usage:
    python3 generate_synthetic.py [--count 1000] [--out /path/to/dir]
"""

import argparse
import os
import random
import re
import sys

SEED = 42

OUT_DIR = "/home/x_x/Programming/LLM/backend/data/raw/synthetic"

# ---------------------------------------------------------------------------
# Thinking / reasoning phrases — drawn from the same style as existing data
# ---------------------------------------------------------------------------
THINKING_PHRASES = [
    "considering what comes next",
    "tracing the logic of this",
    "thinking through the implications",
    "weighing the options here",
    "reasoning about the cause",
    "piecing the details together",
    "recalling what was established",
    "deciding how to frame this",
    "mapping out the structure",
    "checking the assumptions",
    "breaking this down step by step",
    "evaluating the possible approaches",
    "connecting the relevant concepts",
    "identifying the key variables",
    "verifying the intermediate results",
    "considering alternative interpretations",
    "tracing the consequences of this",
    "examining the constraints carefully",
    "formulating a plan of attack",
    "reflecting on what this implies",
    "iterating through the possibilities",
    "testing the hypothesis against the data",
    "working backwards from the desired outcome",
    "comparing the tradeoffs involved",
    "abstracting away the details",
    "generalizing from the specific case",
    "searching for counterexamples",
    "visualizing the problem space",
    "checking for edge cases",
    "validating the reasoning chain",
]

# ---------------------------------------------------------------------------
# Topics – each is a dict with a title and 2-5 paragraph pairs.
# The last paragraph of each topic is the final answer; earlier ones set up
# the problem and work through the reasoning.
# ---------------------------------------------------------------------------

TOPICS = [
    # ---- MATH ----
    {
        "title": "Solving a Quadratic by Factoring",
        "steps": [
            "Solve x^2 - 5x + 6 = 0 by factoring.",
            "We need two numbers that multiply to 6 and add to -5. The pair (-2, -3) works because (-2)*(-3) = 6 and (-2)+(-3) = -5.",
            "So x^2 - 5x + 6 = (x - 2)(x - 3) = 0. Setting each factor to zero: x - 2 = 0 or x - 3 = 0.",
            "The solutions are x = 2 and x = 3.",
        ],
    },
    {
        "title": "Completing the Square",
        "steps": [
            "Solve x^2 + 6x - 7 = 0 by completing the square.",
            "Move the constant: x^2 + 6x = 7. Take half of 6, square it: (6/2)^2 = 9. Add to both sides: x^2 + 6x + 9 = 16.",
            "The left side is (x + 3)^2 = 16. Take square root: x + 3 = ±4.",
            "x = -3 + 4 = 1 or x = -3 - 4 = -7. The solutions are x = 1 and x = -7.",
        ],
    },
    {
        "title": "Quadratic Formula",
        "steps": [
            "Solve 2x^2 - 4x - 3 = 0 using the quadratic formula.",
            "The quadratic formula is x = [-b ± sqrt(b^2 - 4ac)] / (2a). Here a = 2, b = -4, c = -3.",
            "The discriminant: b^2 - 4ac = 16 - 4(2)(-3) = 16 + 24 = 40. sqrt(40) = 2*sqrt(10).",
            "x = [4 ± 2*sqrt(10)] / 4 = [2 ± sqrt(10)] / 2. The two solutions are (2 + sqrt(10))/2 and (2 - sqrt(10))/2.",
        ],
    },
    {
        "title": "Systems of Linear Equations",
        "steps": [
            "Solve the system: 3x + 2y = 12 and 5x - 2y = 4.",
            "Adding the two equations eliminates y: (3x+2y)+(5x-2y) = 12+4 gives 8x = 16, so x = 2.",
            "Substitute x = 2 into the first equation: 3(2)+2y = 12 -> 6+2y = 12 -> 2y = 6 -> y = 3.",
            "The solution is x = 2, y = 3.",
        ],
    },
    {
        "title": "Distance Rate Time Problem",
        "steps": [
            "A train leaves Station A at 60 mph. Two hours later, a second train leaves Station A at 80 mph on the same track. How long will it take the second train to catch the first?",
            "Let t be the time after the second train departs. The first train travels for t+2 hours at 60 mph, covering 60(t+2) miles. The second train travels t hours at 80 mph, covering 80t miles.",
            "When the second catches the first, distances are equal: 60(t+2) = 80t -> 60t + 120 = 80t -> 120 = 20t -> t = 6.",
            "The second train catches the first after 6 hours (8 hours after the first departed).",
        ],
    },
    {
        "title": "Work Problem",
        "steps": [
            "Pipe A can fill a tank in 4 hours. Pipe B can fill the same tank in 6 hours. How long will it take to fill the tank if both pipes are used together?",
            "Pipe A fills at rate 1/4 tank per hour. Pipe B fills at rate 1/6 tank per hour. Combined rate = 1/4 + 1/6 = 3/12 + 2/12 = 5/12 tank per hour.",
            "Time = tank size / rate = 1 / (5/12) = 12/5 = 2.4 hours.",
            "Working together, the pipes fill the tank in 2 hours and 24 minutes.",
        ],
    },
    {
        "title": "Mixture Problem",
        "steps": [
            "How many liters of 20% saline solution must be mixed with 30 liters of 50% saline to get a 40% solution?",
            "Let x be liters of 20% solution. Salt from 20%: 0.2x. Salt from 50%: 0.5 * 30 = 15. Total salt: 0.2x + 15. Total volume: x + 30.",
            "For 40% concentration: (0.2x + 15) / (x + 30) = 0.4 -> 0.2x + 15 = 0.4x + 12 -> 3 = 0.2x -> x = 15.",
            "15 liters of 20% saline solution must be added.",
        ],
    },
    {
        "title": "Area of a Circle",
        "steps": [
            "A circular garden has a radius of 7 meters. What is its area? (Use pi = 22/7)",
            "Area of a circle = pi * r^2. Substituting r = 7: A = (22/7) * 7^2 = (22/7) * 49.",
            "49/7 = 7, so A = 22 * 7 = 154 square meters.",
        ],
    },
    {
        "title": "Pythagorean Theorem",
        "steps": [
            "A right triangle has legs of lengths 5 and 12. What is the length of the hypotenuse?",
            "The Pythagorean theorem states a^2 + b^2 = c^2, where c is the hypotenuse. Here a = 5, b = 12.",
            "c^2 = 5^2 + 12^2 = 25 + 144 = 169. c = sqrt(169) = 13.",
            "The hypotenuse is 13 units long.",
        ],
    },
    {
        "title": "Volume of a Sphere",
        "steps": [
            "What is the volume of a sphere with radius 6 cm? (Use pi = 3.14)",
            "Volume of a sphere = (4/3) * pi * r^3. Substituting r = 6: V = (4/3) * 3.14 * 216.",
            "216 * 3.14 = 678.24. (4/3) * 678.24 = 904.32.",
            "The volume is approximately 904.32 cubic centimeters.",
        ],
    },
    {
        "title": "Slope of a Line",
        "steps": [
            "Find the slope of the line passing through the points (3, 4) and (7, 10).",
            "Slope = (y2 - y1) / (x2 - x1) = (10 - 4) / (7 - 3) = 6 / 4.",
            "6/4 simplifies to 3/2. The slope is 3/2.",
        ],
    },
    {
        "title": "Arithmetic Sequence Sum",
        "steps": [
            "Find the sum of the arithmetic sequence: 3, 7, 11, 15, ... up to 20 terms.",
            "First term a = 3, common difference d = 4, number of terms n = 20. The nth term is a + (n-1)d = 3 + 19*4 = 79.",
            "Sum = n * (first + last) / 2 = 20 * (3 + 79) / 2 = 20 * 82 / 2 = 20 * 41 = 820.",
        ],
    },
    {
        "title": "Geometric Sequence",
        "steps": [
            "Find the 8th term of the geometric sequence: 2, 6, 18, 54, ...",
            "First term a = 2, common ratio r = 3. The nth term is a * r^(n-1).",
            "For n = 8: 2 * 3^7 = 2 * 2187 = 4374.",
        ],
    },
    {
        "title": "Exponential Growth",
        "steps": [
            "A bacteria culture doubles every 3 hours. If there are initially 100 bacteria, how many are there after 24 hours?",
            "Growth formula: N = N0 * 2^(t/d), where N0 = 100, t = 24, d = 3.",
            "N = 100 * 2^(24/3) = 100 * 2^8 = 100 * 256 = 25,600.",
            "After 24 hours, there are 25,600 bacteria.",
        ],
    },
    {
        "title": "Logarithm Evaluation",
        "steps": [
            "Evaluate log_3(81) without a calculator.",
            "We need x such that 3^x = 81. Powers of 3: 3^1 = 3, 3^2 = 9, 3^3 = 27, 3^4 = 81.",
            "Since 3^4 = 81, log_3(81) = 4.",
        ],
    },
    {
        "title": "Factorial Simplification",
        "steps": [
            "Simplify: 10! / 7!",
            "10! = 10 * 9 * 8 * 7!. So 10! / 7! = 10 * 9 * 8 * 7! / 7! = 10 * 9 * 8.",
            "10 * 9 * 8 = 720.",
        ],
    },
    {
        "title": "Combinations",
        "steps": [
            "How many ways can a committee of 3 be chosen from 10 people?",
            "This is a combination: C(10,3) = 10! / (3! * 7!) = (10 * 9 * 8) / (3 * 2 * 1).",
            "720 / 6 = 120. There are 120 possible committees.",
        ],
    },
    {
        "title": "Probability of Independent Events",
        "steps": [
            "What is the probability of rolling a 6 on a fair die and flipping heads on a fair coin?",
            "Rolling a 6 has probability 1/6. Flipping heads has probability 1/2. These are independent events.",
            "P(both) = P(6) * P(heads) = 1/6 * 1/2 = 1/12.",
        ],
    },
    {
        "title": "Expected Value",
        "steps": [
            "A game pays $10 if you roll a 6, $5 if you roll a 5, and nothing otherwise. What is the expected value per roll?",
            "P(6) = 1/6, payout $10. P(5) = 1/6, payout $5. P(other) = 4/6, payout $0.",
            "E = (1/6)*10 + (1/6)*5 + (4/6)*0 = 10/6 + 5/6 = 15/6 = 2.5.",
            "The expected value is $2.50 per roll.",
        ],
    },
    {
        "title": "Standard Deviation",
        "steps": [
            "Find the standard deviation of the dataset: 2, 4, 4, 4, 5, 5, 7, 9.",
            "Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5. Deviations: -3, -1, -1, -1, 0, 0, 2, 4.",
            "Squared deviations: 9, 1, 1, 1, 0, 0, 4, 16. Variance = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4.",
            "Standard deviation = sqrt(4) = 2.",
        ],
    },
    # ---- PHYSICS ----
    {
        "title": "Kinematic Equation",
        "steps": [
            "A car accelerates from rest at 2 m/s^2 for 5 seconds. How far does it travel?",
            "Using s = ut + (1/2)at^2. Initial velocity u = 0, a = 2, t = 5.",
            "s = 0*5 + 1/2 * 2 * 5^2 = 0 + 1 * 25 = 25 meters.",
        ],
    },
    {
        "title": "Free Fall",
        "steps": [
            "A ball is dropped from a height of 45 meters. How long does it take to hit the ground? (g = 10 m/s^2)",
            "Using s = (1/2)gt^2 for free fall from rest. 45 = (1/2) * 10 * t^2 = 5t^2.",
            "t^2 = 45/5 = 9, so t = 3 seconds.",
        ],
    },
    {
        "title": "Newton's Second Law",
        "steps": [
            "A force of 20 N acts on a 5 kg mass. What is the acceleration?",
            "Newton's second law: F = ma. Rearranging: a = F/m.",
            "a = 20 N / 5 kg = 4 m/s^2.",
        ],
    },
    {
        "title": "Friction Force",
        "steps": [
            "A 10 kg box is pulled across a floor with a coefficient of kinetic friction of 0.3. What is the friction force? (g = 9.8 m/s^2)",
            "Normal force N = mg = 10 * 9.8 = 98 N. Friction force f = mu_k * N = 0.3 * 98.",
            "f = 29.4 N.",
        ],
    },
    {
        "title": "Work Done",
        "steps": [
            "How much work is done lifting a 15 kg object vertically 2 meters? (g = 9.8 m/s^2)",
            "Work = force * distance. Force needed = mg = 15 * 9.8 = 147 N.",
            "Work = 147 N * 2 m = 294 J.",
        ],
    },
    {
        "title": "Kinetic Energy",
        "steps": [
            "A 1000 kg car is moving at 20 m/s. What is its kinetic energy?",
            "Kinetic energy = (1/2) * m * v^2 = 0.5 * 1000 * 20^2.",
            "KE = 0.5 * 1000 * 400 = 200,000 J = 200 kJ.",
        ],
    },
    {
        "title": "Potential Energy",
        "steps": [
            "A 2 kg book is on a shelf 3 meters high. What is its gravitational potential energy? (g = 9.8 m/s^2)",
            "Potential energy = mgh = 2 * 9.8 * 3.",
            "PE = 58.8 J.",
        ],
    },
    {
        "title": "Power Calculation",
        "steps": [
            "A motor does 5000 J of work in 10 seconds. What is its power output?",
            "Power = work / time = 5000 J / 10 s.",
            "Power = 500 W.",
        ],
    },
    {
        "title": "Ohm's Law",
        "steps": [
            "A 12 V battery is connected to a 4 ohm resistor. What current flows?",
            "Ohm's law: V = IR. Rearranging: I = V / R = 12 / 4.",
            "I = 3 A.",
        ],
    },
    {
        "title": "Power in a Circuit",
        "steps": [
            "A device draws 2 A at 120 V. What power does it consume?",
            "Electrical power: P = VI = 120 * 2.",
            "P = 240 W.",
        ],
    },
    {
        "title": "Resistors in Series",
        "steps": [
            "Three resistors of 2, 3, and 5 ohms are connected in series. What is the total resistance?",
            "For series: R_total = R1 + R2 + R3 = 2 + 3 + 5.",
            "Total resistance = 10 ohms.",
        ],
    },
    {
        "title": "Resistors in Parallel",
        "steps": [
            "Two resistors of 6 ohms and 3 ohms are connected in parallel. What is the total resistance?",
            "For parallel: 1/R_total = 1/R1 + 1/R2 = 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2.",
            "R_total = 2 ohms.",
        ],
    },
    {
        "title": "Wavelength and Frequency",
        "steps": [
            "A wave has frequency 500 Hz and speed 340 m/s. What is its wavelength?",
            "Wave equation: v = f * lambda. Rearranging: lambda = v / f = 340 / 500.",
            "lambda = 0.68 m.",
        ],
    },
    {
        "title": "Snell's Law",
        "steps": [
            "Light passes from air (n = 1) into water (n = 1.33) at an angle of incidence of 30 degrees. What is the angle of refraction?",
            "Snell's law: n1 * sin(theta1) = n2 * sin(theta2). 1 * sin(30) = 1.33 * sin(theta2).",
            "sin(30) = 0.5, so 0.5 = 1.33 * sin(theta2). sin(theta2) = 0.5/1.33 = 0.376. theta2 = arcsin(0.376) ≈ 22.1 degrees.",
        ],
    },
    {
        "title": "Ideal Gas Law",
        "steps": [
            "2 moles of an ideal gas are at 27 degrees C and 1 atm pressure. What volume does it occupy? (R = 0.0821 L*atm/mol*K)",
            "Ideal gas law: PV = nRT. T = 27 + 273 = 300 K. V = nRT/P = 2 * 0.0821 * 300 / 1.",
            "V = 49.26 L.",
        ],
    },
    {
        "title": "Heat Transfer",
        "steps": [
            "How much heat is needed to raise the temperature of 500 g of water from 20 C to 80 C? (Specific heat of water = 4.18 J/g*C)",
            "Q = mc*delta_T = 500 * 4.18 * (80 - 20) = 500 * 4.18 * 60.",
            "Q = 125,400 J = 125.4 kJ.",
        ],
    },
    {
        "title": "Momentum Conservation",
        "steps": [
            "A 2 kg ball moving at 4 m/s collides with a stationary 1 kg ball. After collision they stick together. What is their velocity?",
            "Conservation of momentum: m1*v1 + m2*v2 = (m1+m2)*v_final. 2*4 + 1*0 = (2+1)*v_final.",
            "8 = 3*v_final, so v_final = 8/3 ≈ 2.67 m/s.",
        ],
    },
    {
        "title": "Centripetal Force",
        "steps": [
            "A 0.5 kg mass swings in a circle of radius 2 m at 3 m/s. What centripetal force is needed?",
            "Centripetal force: F = m*v^2 / r = 0.5 * 3^2 / 2 = 0.5 * 9 / 2.",
            "F = 2.25 N.",
        ],
    },
    {
        "title": "Buoyant Force",
        "steps": [
            "A 0.1 m^3 object is submerged in water (density 1000 kg/m^3). What buoyant force acts on it? (g = 9.8 m/s^2)",
            "Archimedes principle: F_b = density * volume * g = 1000 * 0.1 * 9.8.",
            "F_b = 980 N.",
        ],
    },
    {
        "title": "Simple Pendulum Period",
        "steps": [
            "What is the period of a 1 meter long pendulum? (g = 9.8 m/s^2, pi = 3.14)",
            "Period T = 2*pi * sqrt(L/g) = 2 * 3.14 * sqrt(1/9.8).",
            "sqrt(1/9.8) = sqrt(0.102) = 0.319. T = 6.28 * 0.319 = 2.00 seconds.",
        ],
    },
    # ---- PROGRAMMING / ALGORITHMS ----
    {
        "title": "Finding the Maximum in an Array",
        "steps": [
            "Given the array [3, 7, 2, 9, 5], find the maximum element using a linear scan.",
            "Initialize max = arr[0] = 3. Compare each element: 7 > 3, so max = 7. 2 < 7, skip. 9 > 7, so max = 9. 5 < 9, skip.",
            "The maximum element is 9. This is O(n) time and O(1) space.",
        ],
    },
    {
        "title": "Binary Search",
        "steps": [
            "Search for the value 15 in the sorted array [2, 5, 8, 12, 15, 19, 22] using binary search.",
            "Low = 0, high = 6, mid = 3. arr[3] = 12 < 15, so search right half. Low = 4, high = 6, mid = 5.",
            "arr[5] = 19 > 15, so search left half. Low = 4, high = 4, mid = 4. arr[4] = 15, found it!",
            "Binary search found 15 at index 4 in 3 comparisons.",
        ],
    },
    {
        "title": "Bubble Sort Trace",
        "steps": [
            "Trace one pass of bubble sort on [5, 3, 8, 1, 2].",
            "Compare 5 and 3: swap -> [3, 5, 8, 1, 2]. Compare 5 and 8: no swap. Compare 8 and 1: swap -> [3, 5, 1, 8, 2]. Compare 8 and 2: swap -> [3, 5, 1, 2, 8].",
            "After one pass, the largest element 8 has bubbled to the end. Repeat for remaining n-1 elements.",
        ],
    },
    {
        "title": "Checking for Palindrome",
        "steps": [
            "Determine if 'racecar' is a palindrome.",
            "Use two pointers: left starts at index 0, right at index 6. Compare: r == r, advance. a == a, advance. c == c, advance. e == e, done.",
            "All characters matched. 'racecar' is a palindrome.",
        ],
    },
    {
        "title": "Fibonacci (Recursive)",
        "steps": [
            "Compute the 6th Fibonacci number using recursion, where F(1) = F(2) = 1.",
            "F(6) = F(5) + F(4). F(5) = F(4) + F(3). F(4) = F(3) + F(2). F(3) = F(2) + F(1) = 1 + 1 = 2.",
            "F(4) = 2 + 1 = 3. F(5) = 3 + 2 = 5. F(6) = 5 + 3 = 8.",
        ],
    },
    {
        "title": "Factorial (Iterative)",
        "steps": [
            "Compute 5! iteratively.",
            "Start result = 1. Multiply by 2: result = 2. Multiply by 3: result = 6. Multiply by 4: result = 24. Multiply by 5: result = 120.",
            "5! = 120.",
        ],
    },
    {
        "title": "String Reversal",
        "steps": [
            "Reverse the string 'hello' in place.",
            "Swap first and last: h <-> o -> 'oellh'. Swap second and second-last: e <-> l -> 'olleh'.",
            "The reversed string is 'olleh'.",
        ],
    },
    {
        "title": "Finding Duplicates",
        "steps": [
            "Find duplicate elements in [1, 3, 4, 2, 3, 1, 5] using a hash set.",
            "Initialize empty set. Insert 1, insert 3, insert 4, insert 2. Insert 3 again — 3 already in set, so 3 is a duplicate.",
            "Insert 1 again — 1 already in set, so 1 is a duplicate. Insert 5. Duplicates found: 3 and 1.",
        ],
    },
    {
        "title": "Linked List Reversal",
        "steps": [
            "Reverse the linked list 1 -> 2 -> 3 -> null.",
            "Initialize prev = null, curr = 1. Save next = 2, set curr.next = null. prev = 1, curr = 2.",
            "Save next = 3, set curr.next = 1. prev = 2, curr = 3. Save next = null, set curr.next = 2. prev = 3, curr = null.",
            "New head is 3. Reversed: 3 -> 2 -> 1 -> null.",
        ],
    },
    {
        "title": "Depth-First Search",
        "steps": [
            "Perform DFS on a graph with adjacency list: {A: [B, C], B: [D], C: [E], D: [], E: []}, starting at A.",
            "Visit A, mark visited. Recursively visit neighbor B (A's first neighbor). Visit B, mark visited. Recursively visit neighbor D (B's only neighbor).",
            "Visit D, mark visited. D has no unvisited neighbors, backtrack to B. B has no more unvisited neighbors, backtrack to A.",
            "Visit next unvisited neighbor of A: C. Visit C, mark visited. Visit neighbor E. Visit E, mark visited. DFS order: A, B, D, C, E.",
        ],
    },
    {
        "title": "Breadth-First Search",
        "steps": [
            "Perform BFS on graph {A: [B, C], B: [D], C: [E], D: [], E: []}, starting at A.",
            "Enqueue A. Dequeue A, visit A. Enqueue neighbors: B, C. Dequeue B, visit B. Enqueue D.",
            "Dequeue C, visit C. Enqueue E. Dequeue D, visit D. Dequeue E, visit E.",
            "BFS order: A, B, C, D, E.",
        ],
    },
    {
        "title": "Two Sum Problem",
        "steps": [
            "Given nums = [2, 7, 11, 15] and target = 9, find two numbers that sum to 9.",
            "Use a hash map: iterate. For 2, check if 9-2 = 7 is in map. Not yet, add 2->0. For 7, check if 9-7 = 2 is in map. Yes, at index 0.",
            "Indices [0, 1] with values 2 + 7 = 9.",
        ],
    },
    {
        "title": "FizzBuzz",
        "steps": [
            "Write logic for FizzBuzz up to 15: print Fizz for multiples of 3, Buzz for 5, FizzBuzz for both.",
            "1 -> 1. 2 -> 2. 3 -> Fizz. 4 -> 4. 5 -> Buzz. 6 -> Fizz. 7 -> 7. 8 -> 8. 9 -> Fizz. 10 -> Buzz. 11 -> 11. 12 -> Fizz. 13 -> 13. 14 -> 14. 15 -> FizzBuzz.",
        ],
    },
    {
        "title": "Time Complexity Analysis",
        "steps": [
            "What is the time complexity of nested loops: for i in range(n): for j in range(n): print(i*j)?",
            "The outer loop runs n times. For each outer iteration, the inner loop runs n times. Total operations = n * n = n^2.",
            "The time complexity is O(n^2).",
        ],
    },
    {
        "title": "Count Characters in a String",
        "steps": [
            "Count the frequency of each character in 'hello world'.",
            "h:1, e:1, l:3, o:2, space:1, w:1, r:1, d:1.",
            "The character 'l' appears most frequently with 3 occurrences.",
        ],
    },
    {
        "title": "Leap Year Check",
        "steps": [
            "Is the year 2100 a leap year?",
            "A year is a leap year if divisible by 400, or divisible by 4 but not by 100. 2100 is divisible by 100 but not by 400.",
            "2100 is not a leap year.",
        ],
    },
    {
        "title": "Prime Number Check",
        "steps": [
            "Is 17 a prime number?",
            "Check divisibility up to sqrt(17) ≈ 4. 17 mod 2 = 1, not divisible. 17 mod 3 = 2, not divisible. 17 mod 4 = 1, not divisible.",
            "No divisors found. 17 is a prime number.",
        ],
    },
    {
        "title": "Number of Digits",
        "steps": [
            "How many digits does the number 12345 have?",
            "Repeatedly divide by 10: 12345 / 10 = 1234 (count 1). 1234 / 10 = 123 (count 2). 123 / 10 = 12 (count 3). 12 / 10 = 1 (count 4). 1 / 10 = 0 (count 5).",
            "12345 has 5 digits. Alternatively, use floor(log10(12345)) + 1 = 4 + 1 = 5.",
        ],
    },
    {
        "title": "Fibonacci (Dynamic Programming)",
        "steps": [
            "Compute F(50) efficiently using dynamic programming.",
            "Naive recursion is exponential. Use bottom-up DP: store results in an array. F(1)=1, F(2)=1. For i=3..50: F(i) = F(i-1) + F(i-2).",
            "F(50) = 12,586,269,025. Computed in O(n) time and O(1) space if we only keep the last two values.",
        ],
    },
    # ---- LOGIC PUZZLES ----
    {
        "title": "River Crossing Puzzle",
        "steps": [
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. His boat can only carry him plus one item. The wolf eats the goat if left alone, and the goat eats the cabbage. How does he get everything across?",
            "The farmer cannot leave wolf with goat or goat with cabbage. Take the goat first, leave it on the far side, return alone.",
            "Take the wolf across, but bring the goat back. Leave the goat, take the cabbage across. Return alone, take the goat across.",
            "All items are across safely. The key is bringing the goat back on the second trip to break the conflict pairs.",
        ],
    },
    {
        "title": "The Missing Dollar Riddle",
        "steps": [
            "Three friends pay $30 for a hotel room ($10 each). The manager realizes the room is only $25, so gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each friend. Each friend paid $9, totaling $27, plus the $2 the bellboy kept = $29. Where is the missing dollar?",
            "The calculation is misleading. The $27 total already includes the $2 the bellboy kept ($25 room + $2 kept = $27). Adding the $2 again is wrong.",
            "The $27 plus the $3 returned to the friends = $30. There is no missing dollar; it is a misdirection in the arithmetic.",
        ],
    },
    {
        "title": "The Two Doors Riddle",
        "steps": [
            "You face two doors: one leads to treasure, the other to death. Two guards: one always lies, one always tells truth. You can ask one guard one question. What do you ask?",
            "If you ask 'Which door leads to treasure?', the liar points to the wrong door and the truth-teller to the right one, but you don't know who is who.",
            "Ask either guard: 'What would the other guard say is the treasure door?' Then choose the opposite door. The liar says opposite of truth, truth-teller reports the liar's false answer. Both point to the wrong door.",
        ],
    },
    {
        "title": "The 12 Coin Problem",
        "steps": [
            "You have 12 coins, one is counterfeit (different weight). Using a balance scale only 3 times, find the counterfeit and whether it is heavier or lighter.",
            "Weigh 4 vs 4. If balanced, the fake is among the remaining 4. Use the remaining 2 weighings on those 4 against known-good coins.",
            "If not balanced, the fake is among the 8. Mark coins as potentially heavy or light based on which side went down. Use the next weighings to narrow down to 1 of 3 with known heavy/light status.",
            "With careful planning, the third weighing always identifies the fake and its nature.",
        ],
    },
    {
        "title": "Blue Eyes Puzzle",
        "steps": [
            "On an island of 100 blue-eyed and 100 brown-eyed people, a guru says 'At least one of you has blue eyes.' Everyone sees everyone else's eye color but not their own. No one communicates. What happens?",
            "This is solved by induction. If only 1 person had blue eyes, they would see no one else with blue eyes, deduce they are the one, and leave the first night.",
            "If 2 people have blue eyes, each sees 1 other. If that other doesn't leave the first night, they must be the other blue-eyed person, so both leave the second night.",
            "By induction, all 100 blue-eyed people leave on the 100th night. The guru's statement provides common knowledge that starts the chain.",
        ],
    },
    {
        "title": "Birthday Paradox",
        "steps": [
            "In a room of 23 people, what is the approximate probability that at least two share a birthday?",
            "Compute the complement: the probability that all 23 have different birthdays. P(no match) = (365/365) * (364/365) * (363/365) * ... * (343/365).",
            "P(no match) ≈ 0.4927. So P(at least one match) = 1 - 0.4927 = 0.5073, or about 50.7%.",
            "With just 23 people, there is a greater than 50% chance of a shared birthday.",
        ],
    },
    {
        "title": "The Monty Hall Problem",
        "steps": [
            "Three doors: one car, two goats. You pick door 1. The host (who knows what is behind the doors) opens door 3 revealing a goat. Should you switch to door 2?",
            "Initially, P(car behind your door) = 1/3. P(car behind the other two doors) = 2/3. The host opens a door with a goat, so the 2/3 probability concentrates on the remaining door.",
            "Switching gives a 2/3 chance of winning. Staying gives only 1/3. You should always switch.",
        ],
    },
    {
        "title": "The Chessboard and Rice",
        "steps": [
            "Place 1 grain of rice on the first square of a chessboard, 2 on the second, 4 on the third, doubling each time. How many grains total?",
            "This is the sum of a geometric series: total = 2^0 + 2^1 + ... + 2^63 = 2^64 - 1.",
            "2^64 = 18,446,744,073,709,551,616. Total = 18,446,744,073,709,551,615 grains, or about 18.4 quintillion.",
        ],
    },
    {
        "title": "Handshake Problem",
        "steps": [
            "If 10 people each shake hands with every other person exactly once, how many handshakes occur?",
            "Each person shakes hands with 9 others. That counts each handshake twice (A with B and B with A).",
            "Total = (10 * 9) / 2 = 45 handshakes.",
        ],
    },
    {
        "title": "The Water Jug Problem",
        "steps": [
            "You have a 5-gallon jug and a 3-gallon jug. How do you measure exactly 4 gallons?",
            "Fill the 5-gallon jug. Pour from 5 into 3 until 3 is full, leaving 2 in the 5-gallon jug. Empty the 3-gallon jug.",
            "Pour the remaining 2 gallons from the 5 into the 3. Fill the 5 completely. Pour from 5 into the 3 until the 3 is full (it needs 1 more gallon). That leaves exactly 4 gallons in the 5-gallon jug.",
        ],
    },
    {
        "title": "Clock Angle Problem",
        "steps": [
            "What is the angle between the hour and minute hands at 3:15?",
            "At 3:00, hour hand is at 90 degrees, minute hand at 0 degrees. In 15 minutes, minute hand moves 90 degrees (15/60 * 360). Hour hand moves 7.5 degrees (15/60 * 30 degrees per hour).",
            "Hour hand at 90 + 7.5 = 97.5 degrees. Minute hand at 90 degrees. Angle = |97.5 - 90| = 7.5 degrees.",
        ],
    },
    {
        "title": "The Race Track Problem",
        "steps": [
            "A runner runs a 400-meter track. She runs the first lap at 4 m/s and the second lap at 6 m/s. What is her average speed?",
            "Time for first lap: 400/4 = 100 seconds. Time for second lap: 400/6 = 66.67 seconds. Total distance: 800 meters. Total time: 166.67 seconds.",
            "Average speed = 800/166.67 = 4.8 m/s. Not 5 m/s, because more time is spent at the slower speed.",
        ],
    },
    {
        "title": "The Poisoned Wine",
        "steps": [
            "You have 1000 bottles of wine, one is poisoned. The poison takes 24 hours to show effects. You have 10 test subjects. How do you find the poisoned bottle in 24 hours?",
            "Number each bottle 0 to 999 in binary (10 bits). Each test subject represents one bit position. For each bottle, have the subjects whose bit is 1 drink from it.",
            "After 24 hours, the pattern of dead subjects forms the binary number of the poisoned bottle. 10 subjects can test 2^10 = 1024 bottles, enough for 1000.",
        ],
    },
    # ---- CHEMISTRY ----
    {
        "title": "Balancing Chemical Equations",
        "steps": [
            "Balance: H2 + O2 -> H2O.",
            "Count atoms: left has 2 H and 2 O, right has 2 H and 1 O. Need to balance oxygen. Put coefficient 2 before H2O: H2 + O2 -> 2H2O.",
            "Now right has 4 H and 2 O. Left has 2 H, so put coefficient 2 before H2: 2H2 + O2 -> 2H2O. The equation is balanced.",
        ],
    },
    {
        "title": "Mole Calculation",
        "steps": [
            "How many moles are in 36 grams of water? (H = 1, O = 16)",
            "Molar mass of H2O = 2*1 + 16 = 18 g/mol. Moles = mass / molar mass = 36 / 18.",
            "There are 2 moles of water in 36 grams.",
        ],
    },
    {
        "title": "pH Calculation",
        "steps": [
            "What is the pH of a 0.001 M HCl solution?",
            "HCl is a strong acid that dissociates completely. [H+] = 0.001 M = 10^(-3) M.",
            "pH = -log10[H+] = -log10(10^(-3)) = 3.",
        ],
    },
    {
        "title": "Molarity Calculation",
        "steps": [
            "What is the molarity of a solution made by dissolving 58.5 g of NaCl in enough water to make 500 mL of solution? (Na = 23, Cl = 35.5)",
            "Molar mass of NaCl = 23 + 35.5 = 58.5 g/mol. Moles of NaCl = 58.5 / 58.5 = 1 mole.",
            "Molarity = moles / volume(L) = 1 / 0.5 = 2 M.",
        ],
    },
    {
        "title": "Stoichiometry",
        "steps": [
            "How many grams of CO2 are produced when 24 grams of CH4 burns completely? CH4 + 2O2 -> CO2 + 2H2O. (C=12, H=1, O=16)",
            "Molar mass of CH4 = 12 + 4 = 16 g/mol. Moles of CH4 = 24/16 = 1.5 moles. From the balanced equation, 1 mole CH4 produces 1 mole CO2.",
            "Moles of CO2 = 1.5. Molar mass of CO2 = 12 + 32 = 44 g/mol. Mass of CO2 = 1.5 * 44 = 66 grams.",
        ],
    },
    {
        "title": "Oxidation States",
        "steps": [
            "Determine the oxidation state of manganese in KMnO4.",
            "K has oxidation state +1. O has -4 * 4 = -8 total for oxygen. The compound is neutral: +1 + Mn + (-8) = 0.",
            "Mn = +7. Manganese is in the +7 oxidation state in permanganate.",
        ],
    },
    {
        "title": "Electronegativity Trends",
        "steps": [
            "Arrange F, Cl, Br, I in order of increasing electronegativity.",
            "Electronegativity increases from left to right across a period and decreases down a group. These are all halogens in group 17.",
            "As we go down the group: I < Br < Cl < F. Fluorine is the most electronegative element.",
        ],
    },
    {
        "title": "Gas Stoichiometry",
        "steps": [
            "What volume of O2 at STP is needed to completely react with 10 L of H2? 2H2 + O2 -> 2H2O. At STP, gases have equal mole-volume relationships.",
            "From the balanced equation, 2 moles H2 react with 1 mole O2. The ratio is 2:1 by volume at the same T and P.",
            "10 L H2 requires 5 L O2 at STP.",
        ],
    },
    {
        "title": "Enthalpy Change",
        "steps": [
            "Calculate the enthalpy change when 2 moles of H2O form from H2 and O2. H2 + 1/2 O2 -> H2O, delta_H = -286 kJ/mol.",
            "The reaction as written is for 1 mole of H2O with delta_H = -286 kJ. For 2 moles: delta_H = 2 * (-286) = -572 kJ.",
            "The formation of 2 moles of water releases 572 kJ of heat.",
        ],
    },
    {
        "title": "Half-Life Calculation",
        "steps": [
            "A radioactive isotope has a half-life of 10 years. How much of a 100 g sample remains after 40 years?",
            "After each half-life, the mass halves. After 10 years: 50 g. After 20 years: 25 g. After 30 years: 12.5 g. After 40 years: 6.25 g.",
            "Alternatively: remaining = 100 * (1/2)^(40/10) = 100 * (1/2)^4 = 100 * 1/16 = 6.25 g.",
        ],
    },
    {
        "title": "Limiting Reactant",
        "steps": [
            "If 28 g of N2 reacts with 6 g of H2 to form NH3, what is the limiting reactant? N2 + 3H2 -> 2NH3. (N=14, H=1)",
            "Moles N2 = 28/28 = 1 mol. Moles H2 = 6/2 = 3 mol. The balanced equation requires 1:3 ratio.",
            "1 mol N2 needs exactly 3 mol H2. Both are exactly in stoichiometric proportion. Neither is limiting; both are fully consumed.",
        ],
    },
    {
        "title": "Percent Yield",
        "steps": [
            "A reaction with theoretical yield of 50 g produces only 40 g. What is the percent yield?",
            "Percent yield = (actual yield / theoretical yield) * 100% = (40/50) * 100%.",
            "Percent yield = 80%.",
        ],
    },
    # ---- BIOLOGY ----
    {
        "title": "Mitosis Phases",
        "steps": [
            "List the phases of mitosis in order and describe what happens in each.",
            "Prophase: chromosomes condense, nuclear envelope breaks down. Metaphase: chromosomes align at the metaphase plate. Anaphase: sister chromatids separate to opposite poles. Telophase: nuclear envelopes reform, chromosomes decondense.",
            "Cytokinesis follows mitosis, dividing the cytoplasm to produce two daughter cells.",
        ],
    },
    {
        "title": "DNA Base Pairing",
        "steps": [
            "If one DNA strand has sequence A-T-G-C-C-A, what is the complementary strand?",
            "DNA base pairing rules: A pairs with T, G pairs with C.",
            "The complementary strand is T-A-C-G-G-T.",
        ],
    },
    {
        "title": "Cellular Respiration Summary",
        "steps": [
            "Write the overall equation for aerobic cellular respiration.",
            "Glucose + Oxygen -> Carbon Dioxide + Water + ATP. The balanced equation: C6H12O6 + 6O2 -> 6CO2 + 6H2O + 36-38 ATP.",
            "This process occurs in three stages: glycolysis (cytoplasm), the Krebs cycle (mitochondrial matrix), and the electron transport chain (inner mitochondrial membrane).",
        ],
    },
    {
        "title": "Photosynthesis Equation",
        "steps": [
            "Write the balanced equation for photosynthesis.",
            "Carbon dioxide + Water -> Glucose + Oxygen, using light energy.",
            "6CO2 + 6H2O + light -> C6H12O6 + 6O2. This occurs in the chloroplasts of plant cells.",
        ],
    },
    {
        "title": "Protein Synthesis",
        "steps": [
            "Briefly describe the central dogma of molecular biology.",
            "DNA is transcribed into mRNA in the nucleus. mRNA carries the genetic code to ribosomes in the cytoplasm.",
            "tRNA molecules bring specific amino acids based on mRNA codons. The ribosome links amino acids into a polypeptide chain. This process is called translation.",
        ],
    },
    {
        "title": "Natural Selection",
        "steps": [
            "Explain the three necessary conditions for natural selection to occur.",
            "Variation: individuals in a population differ in traits. Heritability: these differences are at least partly inherited. Differential reproduction: some variants survive and reproduce more successfully than others.",
            "When these conditions hold, the advantageous traits become more common over generations, driving adaptation.",
        ],
    },
    {
        "title": "Blood Types",
        "steps": [
            "What are the four main blood types and which can receive from which?",
            "Type A: A antigens, anti-B antibodies. Type B: B antigens, anti-A antibodies. Type AB: both antigens, no antibodies (universal recipient). Type O: no antigens, both antibodies (universal donor).",
            "O can donate to all. AB can receive from all. A receives from A and O. B receives from B and O.",
        ],
    },
    {
        "title": "Punnett Square",
        "steps": [
            "Two heterozygous pea plants (Tt) are crossed. What are the possible genotypes and phenotypes of the offspring? T = tall, t = short.",
            "Punnett square: Tt x Tt. Gametes: T and t from each parent. Offspring: TT, Tt, tT, tt.",
            "Genotype ratio: 1 TT : 2 Tt : 1 tt. Phenotype ratio: 3 tall : 1 short.",
        ],
    },
    {
        "title": "Enzyme Function",
        "steps": [
            "How do enzymes speed up biochemical reactions?",
            "Enzymes are biological catalysts that lower the activation energy of reactions. Each enzyme has an active site that binds specific substrate molecules.",
            "The induced fit model describes how the enzyme changes shape upon binding, stressing bonds and facilitating the reaction. Enzymes are not consumed and can be reused.",
        ],
    },
    {
        "title": "Food Chain Pyramid",
        "steps": [
            "Why is energy transfer between trophic levels only about 10% efficient?",
            "At each trophic level, energy is lost as heat through respiration, used for cellular processes, or egested as waste. Only the energy incorporated into biomass is available to the next level.",
            "This explains why food chains rarely have more than 4-5 levels and why top predators are rare compared to producers.",
        ],
    },
    # ---- ECONOMICS ----
    {
        "title": "Supply and Demand",
        "steps": [
            "If the price of coffee increases, what happens to the demand for tea?",
            "Coffee and tea are substitute goods. When the price of coffee rises, consumers switch to the relatively cheaper alternative.",
            "The demand for tea increases, shifting its demand curve to the right, leading to higher equilibrium price and quantity for tea.",
        ],
    },
    {
        "title": "Price Elasticity of Demand",
        "steps": [
            "A 10% price increase causes a 20% decrease in quantity demanded. What is the price elasticity of demand?",
            "PED = % change in quantity demanded / % change in price = -20% / +10% = -2.",
            "|PED| = 2 > 1, so demand is elastic. Revenue will fall when price increases because the quantity drop is proportionally larger.",
        ],
    },
    {
        "title": "Opportunity Cost",
        "steps": [
            "You spend $50 on a concert ticket and could have used that money and time to work 4 hours at $15/hour. What is the opportunity cost?",
            "Explicit cost: $50 for the ticket. Implicit cost: the foregone wages of 4 * $15 = $60.",
            "Opportunity cost = explicit + implicit = $50 + $60 = $110.",
        ],
    },
    {
        "title": "Comparative Advantage",
        "steps": [
            "Country A can produce 10 cars or 5 computers per worker. Country B can produce 8 cars or 4 computers per worker. Should they trade?",
            "A's opportunity cost of 1 car = 0.5 computers. B's opportunity cost of 1 car = 0.5 computers. Both have the same opportunity cost ratio.",
            "No comparative advantage exists. Trade does not provide gains since the ratios are identical.",
        ],
    },
    {
        "title": "Inflation Calculation",
        "steps": [
            "If a basket of goods costs $100 in 2020 and $110 in 2021, what is the inflation rate?",
            "Inflation rate = ((CPI_new - CPI_old) / CPI_old) * 100% = ((110 - 100) / 100) * 100%.",
            "Inflation rate = 10%.",
        ],
    },
    {
        "title": "Compound Interest",
        "steps": [
            "If you invest $5000 at 6% annual interest compounded monthly, what is the value after 2 years?",
            "A = P * (1 + r/n)^(nt) = 5000 * (1 + 0.06/12)^(12*2) = 5000 * (1.005)^24.",
            "(1.005)^24 = 1.12716. A = 5000 * 1.12716 = $5635.80.",
        ],
    },
    {
        "title": "GDP Calculation",
        "steps": [
            "What are the four components of GDP expenditure?",
            "GDP = C + I + G + NX, where C = consumption (household spending), I = investment (business spending on capital), G = government spending, and NX = net exports (exports - imports).",
            "Consumption is typically the largest component in developed economies.",
        ],
    },
    {
        "title": "Law of Diminishing Marginal Utility",
        "steps": [
            "Explain why the first slice of pizza is more satisfying than the fifth.",
            "Marginal utility is the additional satisfaction from consuming one more unit. As consumption increases, each additional unit provides less additional satisfaction.",
            "This explains the downward-sloping demand curve: consumers only buy more at lower prices because the marginal utility of each additional unit decreases.",
        ],
    },
    {
        "title": "Market Equilibrium",
        "steps": [
            "Demand is Qd = 100 - 2P and supply is Qs = 3P. Find the equilibrium price and quantity.",
            "At equilibrium, Qd = Qs. 100 - 2P = 3P -> 100 = 5P -> P = 20.",
            "Q = 3 * 20 = 60. Equilibrium price is $20, equilibrium quantity is 60 units.",
        ],
    },
    {
        "title": "Consumer Surplus",
        "steps": [
            "If demand is P = 50 - Q and price is $30, what is the consumer surplus?",
            "At P = 30, Q = 50 - 30 = 20. The demand curve represents willingness to pay. Consumer surplus is the area below demand and above price, up to Q = 20.",
            "This is a triangle with base 20 and height (50 - 30) = 20. CS = 1/2 * 20 * 20 = 200.",
        ],
    },
    # ---- LANGUAGE / LINGUISTICS ----
    {
        "title": "Parts of Speech",
        "steps": [
            "Identify the parts of speech in: 'The quick brown fox jumps over the lazy dog.'",
            "The (article), quick (adjective), brown (adjective), fox (noun), jumps (verb), over (preposition), the (article), lazy (adjective), dog (noun).",
            "The sentence contains 1 verb, 1 noun, 3 adjectives, 1 preposition, and 2 articles.",
        ],
    },
    {
        "title": "Sentence Structure",
        "steps": [
            "Identify the subject, verb, and object in: 'The cat chased the mouse.'",
            "Subject: The cat (performer of the action). Verb: chased (the action). Object: the mouse (receiver of the action).",
            "This is a simple SVO (Subject-Verb-Object) sentence structure.",
        ],
    },
    {
        "title": "Prefix and Suffix",
        "steps": [
            "Break down the word 'unbelievable' into its morphemes.",
            "un- (prefix meaning 'not') + believe (root word) + -able (suffix meaning 'capable of').",
            "'Unbelievable' means 'not capable of being believed.'",
        ],
    },
    {
        "title": "Active vs Passive Voice",
        "steps": [
            "Convert 'The ball was thrown by the boy' from passive to active voice.",
            "In passive voice, the subject receives the action. To convert, make the agent (the boy) the subject, and the receiver (the ball) the object.",
            "Active voice: 'The boy threw the ball.'",
        ],
    },
    {
        "title": "Synonyms and Antonyms",
        "steps": [
            "Find a synonym and antonym for 'benevolent'.",
            "Synonym: kind, generous, charitable. Antonym: malevolent, cruel, unkind.",
            "Benevolent means well-meaning and kindly, from Latin bene (well) + volent (wishing).",
        ],
    },
    {
        "title": "Common Etymologies",
        "steps": [
            "What is the origin of the word 'etymology' itself?",
            "From Greek 'etymon' meaning 'true sense' and 'logos' meaning 'word' or 'study'. So etymology is the study of the true origin of words.",
            "It entered English through Old French 'etimologie' in the 14th century.",
        ],
    },
    {
        "title": "Figurative Language",
        "steps": [
            "Identify the figure of speech: 'The world is a stage.'",
            "This is a metaphor — a direct comparison between two unlike things without using 'like' or 'as'. The world is not literally a stage, but is being compared to one.",
            "If it said 'The world is like a stage', it would be a simile. This quote is from Shakespeare's As You Like It.",
        ],
    },
    {
        "title": "Grammar: Subject-Verb Agreement",
        "steps": [
            "Which is correct: 'The group of students is/are meeting'?",
            "The subject is 'group' (singular), not 'students' (object of preposition). 'Of students' is a prepositional phrase modifying 'group'.",
            "Correct: 'The group of students is meeting.' The singular verb agrees with the singular subject.",
        ],
    },
    {
        "title": "Phonetics: Voiced vs Voiceless",
        "steps": [
            "Are the sounds /b/ and /p/ voiced or voiceless?",
            "/b/ is voiced — the vocal cords vibrate when producing it. /p/ is voiceless — the vocal cords do not vibrate.",
            "Both are bilabial plosives, differing only in voicing. This is a minimal pair in English (e.g., 'bat' vs 'pat').",
        ],
    },
    {
        "title": "Semantic Field",
        "steps": [
            "Identify the semantic field of: surgeon, scalpel, stethoscope, ward, diagnosis.",
            "All words relate to medicine or healthcare. Surgeon is a medical professional, scalpel and stethoscope are medical instruments, ward is a hospital unit, diagnosis is a medical assessment.",
            "These words share a common semantic field, allowing them to co-occur in medical contexts.",
        ],
    },
    # ---- PHILOSOPHY / LOGIC ----
    {
        "title": "Deductive vs Inductive Reasoning",
        "steps": [
            "Identify the type of reasoning: 'All humans are mortal. Socrates is human. Therefore, Socrates is mortal.'",
            "This is deductive reasoning: the conclusion necessarily follows from the premises. If the premises are true, the conclusion must be true.",
            "In deductive reasoning, the conclusion is guaranteed. In inductive reasoning, the conclusion is probable but not guaranteed.",
        ],
    },
    {
        "title": "Logical Fallacy: Straw Man",
        "steps": [
            "Identify the fallacy: 'Person A says we should improve public schools. Person B responds: So you want to raise taxes and bankrupt everyone?'",
            "This is a straw man fallacy. Person B misrepresents Person A's position as an extreme version, then attacks that distorted version.",
            "The straw man is a common rhetorical tactic. The correct response is to point out the misrepresentation and restate the original position accurately.",
        ],
    },
    {
        "title": "The Trolley Problem",
        "steps": [
            "A trolley is heading toward five people. You can pull a lever to divert it onto a side track where it kills one person. Do you pull the lever?",
            "Utilitarian approach: pull the lever to save five at the cost of one, maximizing overall well-being. Deontological approach: actively causing a death is wrong, so do not intervene.",
            "The trolley problem highlights the tension between consequentialist and duty-based ethics. There is no universally agreed answer.",
        ],
    },
    {
        "title": "Cogito Ergo Sum",
        "steps": [
            "What is Descartes' famous philosophical statement and what does it mean?",
            "'Cogito, ergo sum' — 'I think, therefore I am.' Descartes used this as the foundation of his epistemology after methodical doubt.",
            "Even if an evil demon deceives me about everything, the act of being deceived or thinking requires a thinker. Thus, the existence of the thinking self is indubitable.",
        ],
    },
    {
        "title": "Plato's Allegory of the Cave",
        "steps": [
            "Summarize Plato's Allegory of the Cave.",
            "Prisoners chained in a cave see only shadows on the wall, believing these are reality. One prisoner escapes, sees the actual world, and realizes the shadows were mere projections.",
            "The allegory represents the philosopher's journey from ignorance (shadows) to knowledge (the Forms illuminated by the Good). It illustrates the difficulty of enlightenment and the resistance of those still in the cave.",
        ],
    },
    {
        "title": "The Problem of Evil",
        "steps": [
            "What is the logical problem of evil?",
            "If God is omnipotent (all-powerful), omniscient (all-knowing), and omnibenevolent (all-good), then evil should not exist. But evil does exist. Therefore, such a God cannot exist.",
            "Responses include the free will defense (evil results from human free choice), the soul-making theodicy (evil builds character), and the epistemic distance argument (God must be somewhat hidden).",
        ],
    },
    {
        "title": "Occam's Razor",
        "steps": [
            "Explain Occam's Razor and its use in reasoning.",
            "Occam's Razor states that among competing hypotheses, the one with the fewest assumptions should be selected. It is a heuristic, not a proof.",
            "Also formulated as 'Entities should not be multiplied without necessity.' It guides scientific theory choice toward simpler explanations.",
        ],
    },
    {
        "title": "The Ship of Theseus",
        "steps": [
            "If all parts of a ship are replaced over time, is it still the same ship?",
            "This is a paradox about identity over time. If the original planks are reassembled, which vessel is the original — the continuously replaced one or the reassembled one?",
            "The paradox explores whether identity is tied to material composition, continuity, or form. It applies to personal identity (is a person with all new cells still the same person?).",
        ],
    },
    {
        "title": "A Priori vs A Posteriori",
        "steps": [
            "Classify: 'All bachelors are unmarried' as a priori or a posteriori.",
            "A priori knowledge is independent of experience. A posteriori depends on empirical observation. This statement is true by definition — the concept 'bachelor' includes 'unmarried'.",
            "It is a priori, analytic knowledge: true solely by virtue of the meaning of the terms.",
        ],
    },
    {
        "title": "The Veil of Ignorance",
        "steps": [
            "Explain Rawls' Veil of Ignorance.",
            "Rawls asks us to design a just society from behind a veil where we do not know our own position in that society (our wealth, talents, gender, etc.).",
            "From this impartial perspective, rational people would choose principles that protect the worst-off: the difference principle. This yields a conception of justice as fairness.",
        ],
    },
    # ---- GEOGRAPHY / GENERAL ----
    {
        "title": "Latitude and Longitude",
        "steps": [
            "What is the significance of the Prime Meridian?",
            "The Prime Meridian (0 degrees longitude) runs through Greenwich, England. It divides the Earth into Eastern and Western hemispheres.",
            "It was established as the international standard in 1884. Longitude is measured east or west from this line up to 180 degrees.",
        ],
    },
    {
        "title": "Population Density",
        "steps": [
            "A city has 500,000 people and an area of 200 square kilometers. What is the population density?",
            "Population density = population / area = 500,000 / 200.",
            "Density = 2,500 people per square kilometer.",
        ],
    },
    {
        "title": "Scale Factor",
        "steps": [
            "On a map with scale 1:100,000, two cities are 5 cm apart. What is the actual distance?",
            "Scale 1:100,000 means 1 cm on the map represents 100,000 cm on the ground. 5 cm represents 500,000 cm.",
            "500,000 cm = 5,000 m = 5 km. The actual distance is 5 kilometers.",
        ],
    },
    {
        "title": "Temperature Conversion",
        "steps": [
            "Convert 25 degrees Celsius to Fahrenheit.",
            "Formula: F = C * 9/5 + 32 = 25 * 9/5 + 32 = 45 + 32.",
            "25 C = 77 F.",
        ],
    },
    {
        "title": "Significant Figures",
        "steps": [
            "How many significant figures in 0.00450?",
            "Leading zeros are not significant. Trailing zeros after the decimal are significant. Digits: 4, 5, and the trailing 0.",
            "0.00450 has 3 significant figures.",
        ],
    },
    {
        "title": "Scientific Notation",
        "steps": [
            "Write 0.000000123 in scientific notation.",
            "Move the decimal 7 places to the right to get 1.23. The exponent is -7 because we moved right.",
            "0.000000123 = 1.23 * 10^(-7).",
        ],
    },
    {
        "title": "Percentage Change",
        "steps": [
            "A stock price drops from $80 to $60. What is the percentage decrease?",
            "Percentage change = (new - old) / old * 100% = (60 - 80) / 80 * 100% = (-20) / 80 * 100%.",
            "Percentage decrease = -25%, or a 25% decrease.",
        ],
    },
    {
        "title": "Simple Interest",
        "steps": [
            "You borrow $2000 at 5% simple interest per year for 3 years. What is the total interest?",
            "Simple interest = P * r * t = 2000 * 0.05 * 3.",
            "Interest = $300. Total repayment = $2,300.",
        ],
    },
    {
        "title": "Rate of Change",
        "steps": [
            "A plant grows from 10 cm to 25 cm over 5 weeks. What is the average rate of growth per week?",
            "Average rate of change = (change in height) / (change in time) = (25 - 10) / 5.",
            "Average growth rate = 15/5 = 3 cm per week.",
        ],
    },
    {
        "title": "Ratio and Proportion",
        "steps": [
            "If 3 apples cost $1.50, how much do 7 apples cost?",
            "Unit price = $1.50 / 3 = $0.50 per apple. Cost of 7 apples = 7 * $0.50.",
            "7 apples cost $3.50.",
        ],
    },
    # ---- MORE MATH ----
    {
        "title": "Derivative of a Polynomial",
        "steps": [
            "Find the derivative of f(x) = 3x^2 + 2x - 5.",
            "Using power rule: derivative of 3x^2 is 3*2x = 6x. Derivative of 2x is 2. Derivative of -5 (constant) is 0.",
            "f'(x) = 6x + 2.",
        ],
    },
    {
        "title": "Integration",
        "steps": [
            "Evaluate the indefinite integral of 2x dx.",
            "The integral of 2x is x^2 + C, using the power rule in reverse. Check by differentiating: d/dx (x^2 + C) = 2x.",
            "Integral of 2x dx = x^2 + C.",
        ],
    },
    {
        "title": "Definite Integral",
        "steps": [
            "Evaluate the definite integral of x from 0 to 4.",
            "The antiderivative of x is x^2/2. Evaluate at bounds: (4^2/2) - (0^2/2) = 16/2 - 0 = 8.",
            "The area under the curve y = x from x = 0 to x = 4 is 8.",
        ],
    },
    {
        "title": "Chain Rule",
        "steps": [
            "Find the derivative of h(x) = (3x^2 + 1)^5.",
            "Let u = 3x^2 + 1, then h = u^5. Chain rule: dh/dx = dh/du * du/dx = 5u^4 * 6x.",
            "h'(x) = 5(3x^2 + 1)^4 * 6x = 30x(3x^2 + 1)^4.",
        ],
    },
    {
        "title": "Limits",
        "steps": [
            "Evaluate the limit as x approaches 2 of (x^2 - 4)/(x - 2).",
            "Direct substitution gives 0/0, indeterminate. Factor numerator: (x-2)(x+2)/(x-2). Cancel (x-2).",
            "Limit = x + 2 as x -> 2 = 4.",
        ],
    },
    {
        "title": "Absolute Value Equation",
        "steps": [
            "Solve |x - 3| = 5.",
            "Absolute value means x - 3 = 5 or x - 3 = -5.",
            "x = 8 or x = -2.",
        ],
    },
    {
        "title": "Inequality Solving",
        "steps": [
            "Solve 2x - 3 < 7.",
            "Add 3 to both sides: 2x < 10. Divide by 2: x < 5.",
            "The solution is x < 5, or in interval notation: (-infinity, 5).",
        ],
    },
    {
        "title": "Midpoint Formula",
        "steps": [
            "Find the midpoint between (2, 3) and (8, 7).",
            "Midpoint = ((x1+x2)/2, (y1+y2)/2) = ((2+8)/2, (3+7)/2) = (10/2, 10/2).",
            "Midpoint = (5, 5).",
        ],
    },
    {
        "title": "Distance Formula",
        "steps": [
            "Find the distance between (1, 2) and (4, 6).",
            "Distance = sqrt((x2-x1)^2 + (y2-y1)^2) = sqrt((4-1)^2 + (6-2)^2) = sqrt(3^2 + 4^2).",
            "Distance = sqrt(9 + 16) = sqrt(25) = 5.",
        ],
    },
    {
        "title": "Equation of a Circle",
        "steps": [
            "Write the equation of a circle with center (3, -2) and radius 4.",
            "The standard form is (x - h)^2 + (y - k)^2 = r^2, where (h,k) is the center.",
            "(x - 3)^2 + (y + 2)^2 = 16.",
        ],
    },
    {
        "title": "Direct Variation",
        "steps": [
            "If y varies directly with x and y = 12 when x = 4, find y when x = 7.",
            "Direct variation: y = kx. From 12 = k*4, k = 3. So y = 3x.",
            "When x = 7: y = 3 * 7 = 21.",
        ],
    },
    {
        "title": "Inverse Variation",
        "steps": [
            "If y varies inversely with x and y = 6 when x = 8, find y when x = 12.",
            "Inverse variation: y = k/x. From 6 = k/8, k = 48. So y = 48/x.",
            "When x = 12: y = 48/12 = 4.",
        ],
    },
    {
        "title": "Sequence: Next Term",
        "steps": [
            "Find the next term in the sequence: 1, 1, 2, 3, 5, 8, 13, ...",
            "This is the Fibonacci sequence: each term is the sum of the two previous terms. 8 + 5 = 13, so the next is 13 + 8 = 21.",
            "The next term is 21.",
        ],
    },
    {
        "title": "Unit Conversion: Metric",
        "steps": [
            "Convert 2500 meters to kilometers.",
            "1 km = 1000 m. 2500 / 1000 = 2.5.",
            "2500 m = 2.5 km.",
        ],
    },
    {
        "title": "Unit Conversion: Imperial",
        "steps": [
            "Convert 3 miles to feet.",
            "1 mile = 5280 feet. 3 * 5280 = 15,840.",
            "3 miles = 15,840 feet.",
        ],
    },
    {
        "title": "Perimeter of a Rectangle",
        "steps": [
            "A rectangle has length 12 cm and width 8 cm. What is its perimeter?",
            "Perimeter = 2 * (length + width) = 2 * (12 + 8) = 2 * 20.",
            "Perimeter = 40 cm.",
        ],
    },
    {
        "title": "Area of a Triangle",
        "steps": [
            "A triangle has base 10 cm and height 6 cm. What is its area?",
            "Area = 1/2 * base * height = 1/2 * 10 * 6.",
            "Area = 30 square cm.",
        ],
    },
    {
        "title": "Surface Area of a Cube",
        "steps": [
            "A cube has side length 5 cm. What is its surface area?",
            "Each face has area 5*5 = 25 square cm. A cube has 6 faces.",
            "Surface area = 6 * 25 = 150 square cm.",
        ],
    },
    {
        "title": "Volume of a Rectangular Prism",
        "steps": [
            "A box measures 3 m by 4 m by 5 m. What is its volume?",
            "Volume = length * width * height = 3 * 4 * 5.",
            "Volume = 60 cubic meters.",
        ],
    },
    {
        "title": "Circumference of a Circle",
        "steps": [
            "A circle has diameter 14 cm. What is its circumference? (Use pi = 22/7)",
            "Circumference = pi * d = (22/7) * 14 = 22 * 2.",
            "Circumference = 44 cm.",
        ],
    },
    # ---- MORE PHYSICS ----
    {
        "title": "Pressure Calculation",
        "steps": [
            "A force of 100 N is applied over an area of 0.5 m^2. What is the pressure?",
            "Pressure = force / area = 100 / 0.5.",
            "Pressure = 200 Pa.",
        ],
    },
    {
        "title": "Density",
        "steps": [
            "An object has mass 300 g and volume 150 cm^3. What is its density?",
            "Density = mass / volume = 300 / 150.",
            "Density = 2 g/cm^3.",
        ],
    },
    {
        "title": "Speed of Sound",
        "steps": [
            "Thunder is heard 3 seconds after a lightning flash. How far away is the storm? (Speed of sound = 343 m/s)",
            "Distance = speed * time = 343 * 3.",
            "Distance = 1029 meters, approximately 1 km.",
        ],
    },
    {
        "title": "Refraction Index",
        "steps": [
            "Light travels at 2.25 * 10^8 m/s in a certain material. What is the refractive index? (c = 3 * 10^8 m/s)",
            "Refractive index n = c / v = (3 * 10^8) / (2.25 * 10^8).",
            "n = 1.33, which matches the refractive index of water.",
        ],
    },
    {
        "title": "Specific Heat Capacity",
        "steps": [
            "500 J of heat raises the temperature of 100 g of a substance by 10 degrees C. What is its specific heat capacity?",
            "Q = mc*delta_T. c = Q / (m * delta_T) = 500 / (0.1 * 10) = 500 / 1.",
            "c = 500 J/(kg*K).",
        ],
    },
    {
        "title": "Coulomb's Law",
        "steps": [
            "Two charges of 2 * 10^(-6) C each are separated by 0.3 m. What is the electrostatic force between them? (k = 9 * 10^9 N*m^2/C^2)",
            "F = k * q1 * q2 / r^2 = (9*10^9) * (2*10^(-6))^2 / (0.3)^2.",
            "F = (9*10^9) * (4*10^(-12)) / 0.09 = 36*10^(-3) / 0.09 = 0.4 N.",
        ],
    },
    {
        "title": "Doppler Effect",
        "steps": [
            "A source emitting 500 Hz approaches a stationary observer at 30 m/s. What frequency does the observer hear? (v = 340 m/s)",
            "Doppler formula: f' = f * v / (v - v_source) = 500 * 340 / (340 - 30) = 500 * 340 / 310.",
            "f' = 500 * 1.097 = 548.4 Hz. The observer hears a higher pitch.",
        ],
    },
    {
        "title": "Momentum Impulse",
        "steps": [
            "A 0.5 kg ball moving at 10 m/s is stopped in 0.1 s. What average force is applied?",
            "Impulse = change in momentum = 0.5 * 10 = 5 N*s. Force = impulse / time = 5 / 0.1.",
            "Average force = 50 N.",
        ],
    },
    {
        "title": "Rotational Kinetic Energy",
        "steps": [
            "A disk with moment of inertia 0.02 kg*m^2 rotates at 10 rad/s. What is its rotational kinetic energy?",
            "Rotational KE = 1/2 * I * omega^2 = 0.5 * 0.02 * 10^2 = 0.5 * 0.02 * 100.",
            "Rotational KE = 1 J.",
        ],
    },
    {
        "title": "Thermal Expansion",
        "steps": [
            "A steel rail is 20 m long at 10 C. How much does it expand when the temperature reaches 40 C? (alpha = 1.2 * 10^(-5) / C)",
            "delta_L = L0 * alpha * delta_T = 20 * 1.2*10^(-5) * 30 = 20 * 3.6*10^(-4).",
            "delta_L = 0.0072 m = 7.2 mm. Expansion gaps must accommodate this.",
        ],
    },
    # ---- MORE PROGRAMMING ----
    {
        "title": "Stack Operations",
        "steps": [
            "Evaluate the postfix expression: 5 3 + 8 *.",
            "Push 5. Push 3. Pop 3 and 5, compute 5+3 = 8, push 8. Push 8. Pop 8 and 8, compute 8*8 = 64. Push 64.",
            "Result: 64.",
        ],
    },
    {
        "title": "Queue vs Stack",
        "steps": [
            "What is the difference between a stack and a queue?",
            "Stack: LIFO (Last In, First Out). Push and pop operations at the top. Queue: FIFO (First In, First Out). Enqueue at rear, dequeue at front.",
            "A stack is like a pile of plates; a queue is like a line of people.",
        ],
    },
    {
        "title": "Recursive vs Iterative",
        "steps": [
            "Compare recursion and iteration for computing factorial.",
            "Recursive: return n * factorial(n-1) with base case n=1. Iterative: loop from 1 to n, multiplying. Both compute the same result.",
            "Recursion is elegantly declarative but may cause stack overflow for large n. Iteration uses constant stack space.",
        ],
    },
    {
        "title": "Hash Table Collision",
        "steps": [
            "Explain one method of handling hash table collisions.",
            "Chaining: each bucket stores a linked list of key-value pairs. On insertion, append to the list at the hashed index. On lookup, traverse the list to find the key.",
            "Open addressing: on collision, probe the next available slot. Linear probing checks consecutive slots; quadratic probing checks slots at increasing intervals.",
        ],
    },
    {
        "title": "Big-O for Common Operations",
        "steps": [
            "What is the time complexity of inserting at the beginning of an array-backed list vs a linked list?",
            "Array-backed list: O(n) because elements must shift right. Linked list: O(1) because we only update the head pointer.",
            "Linked lists excel at head insertions/deletions but have O(n) random access. Arrays excel at random access (O(1)).",
        ],
    },
    {
        "title": "Selection Sort",
        "steps": [
            "Trace selection sort on [29, 10, 14, 37, 13].",
            "Find minimum (10) at index 1, swap with index 0: [10, 29, 14, 37, 13]. Find minimum in [1..4] (13 at index 4), swap with index 1: [10, 13, 14, 37, 29].",
            "Find minimum in [2..4] (14 at index 2), already in place. Find minimum in [3..4] (29 at index 4), swap with index 3: [10, 13, 14, 29, 37]. Sorted.",
        ],
    },
    {
        "title": "Insertion Sort",
        "steps": [
            "Trace insertion sort on [8, 3, 5, 1].",
            "Start with [8]. Insert 3: compare with 8, shift 8 right: [3, 8, 5, 1]. Insert 5: compare with 8, shift 8 right, compare with 3, insert: [3, 5, 8, 1].",
            "Insert 1: compare with 8, shift. Compare with 5, shift. Compare with 3, shift. Insert: [1, 3, 5, 8]. Sorted.",
        ],
    },
    {
        "title": "Merge Sort Concept",
        "steps": [
            "Explain the divide step of merge sort on [38, 27, 43, 3, 9, 82, 10].",
            "Split into [38, 27, 43, 3] and [9, 82, 10]. Split first half: [38, 27] and [43, 3]. Split further: [38], [27], [43], [3]. Split second half: [9, 82] and [10], then [9], [82], [10].",
            "Now recursively merge back in sorted order. Merge sort is O(n log n) in all cases.",
        ],
    },
    {
        "title": "Tree Traversal: Inorder",
        "steps": [
            "Given a binary tree with root 4, left subtree rooted at 2 with leaves 1 and 3, right subtree rooted at 5 with right child 6. List the inorder traversal.",
            "Inorder visits left subtree, then root, then right subtree. Left subtree of 4: inorder of [2 with 1,3] = 1, 2, 3. Then root 4.",
            "Right subtree of 4: inorder of [5 with right 6] = 5, 6. Full inorder: 1, 2, 3, 4, 5, 6.",
        ],
    },
    {
        "title": "Bubble Sort Optimization",
        "steps": [
            "How can basic bubble sort be optimized?",
            "Track whether any swaps occurred in a pass. If no swaps, the array is already sorted and we can terminate early. This makes best-case O(n).",
            "Also, after each pass, the largest remaining element is at its final position, so the inner loop can shrink by one each iteration.",
        ],
    },
    # ---- MORE LOGIC ----
    {
        "title": "Syllogism Validity",
        "steps": [
            "Is this syllogism valid? All mammals have fur. All dogs are mammals. Therefore, all dogs have fur.",
            "Premise 1: All mammals have fur (All M are F). Premise 2: All dogs are mammals (All D are M). Conclusion: All dogs have fur (All D are F).",
            "This is a valid syllogism. If the premises are true, the conclusion must be true. It is an example of the Barbara syllogism form.",
        ],
    },
    {
        "title": "Confirmation Bias",
        "steps": [
            "Define confirmation bias.",
            "Confirmation bias is the tendency to search for, interpret, and recall information that confirms one's preexisting beliefs while giving less weight to contradictory evidence.",
            "It is a common cognitive bias that can lead to poor decision-making. Mitigation strategies include actively seeking disconfirming evidence and considering alternative hypotheses.",
        ],
    },
    {
        "title": "Gambler's Fallacy",
        "steps": [
            "What is the gambler's fallacy?",
            "The gambler's fallacy is the belief that if an event has occurred more frequently than usual, it is less likely to occur in the future (or vice versa), when events are independent.",
            "Example: after 5 coin flips landing heads, believing tails is 'due.' In reality, each flip has a 50% chance regardless of past outcomes.",
        ],
    },
    {
        "title": "The Halo Effect",
        "steps": [
            "Explain the halo effect with an example.",
            "The halo effect is a cognitive bias where a positive impression in one area influences opinion in another area. For example, assuming an attractive person is also intelligent and kind.",
            "This bias affects hiring decisions, performance evaluations, and marketing. Being aware of it helps reduce its influence.",
        ],
    },
    {
        "title": "Base Rate Fallacy",
        "steps": [
            "A test for a disease is 99% accurate. If 1% of the population has the disease and you test positive, what is the probability you actually have it?",
            "In 10,000 people, 100 have the disease (1%). 99 of these test positive (99% sensitivity). Of the 9,900 without the disease, 99 test positive (1% false positive rate).",
            "Total positives = 99 + 99 = 198. P(disease | positive) = 99/198 = 50%. Ignoring the base rate (1% prevalence) would lead to overestimating this probability.",
        ],
    },
    {
        "title": "Correlation vs Causation",
        "steps": [
            "Explain the difference between correlation and causation with an example.",
            "Correlation means two variables move together. Causation means one variable directly affects the other. Correlation does not imply causation.",
            "Example: ice cream sales and drowning deaths are correlated (both increase in summer), but ice cream does not cause drowning. The confounding variable is hot weather.",
        ],
    },
    {
        "title": "The Sunk Cost Fallacy",
        "steps": [
            "What is the sunk cost fallacy?",
            "Sunk cost fallacy is the tendency to continue investing in a losing endeavor because of past unrecoverable investments (time, money, effort), rather than cutting losses.",
            "Rational decision-making should ignore sunk costs and only consider future costs and benefits. Example: continuing a bad movie because you already paid for the ticket.",
        ],
    },
    {
        "title": "Anchoring Bias",
        "steps": [
            "Define anchoring bias and give an example.",
            "Anchoring bias is the tendency to rely too heavily on the first piece of information encountered (the anchor) when making decisions.",
            "Example: in salary negotiations, the first offer sets an anchor. A higher initial offer leads to higher final settlements, even if the anchor is arbitrary.",
        ],
    },
    {
        "title": "Tversky and Kahneman's Linda Problem",
        "steps": [
            "Linda is 31, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with social justice. Which is more probable: Linda is a bank teller, or Linda is a bank teller and active in the feminist movement?",
            "The conjunction fallacy: the conjunction of two events cannot be more probable than either event alone. P(bank teller AND feminist) <= P(bank teller).",
            "Yet most people choose the conjunction because it is more representative. This demonstrates how intuitive reasoning violates probability laws.",
        ],
    },
    {
        "title": "The Framing Effect",
        "steps": [
            "Explain the framing effect using the example of medical treatment.",
            "The framing effect: people react differently to the same information depending on how it is presented. Surgery with 90% survival rate is chosen more often than surgery with 10% mortality rate.",
            "The information is identical but differently framed. This shows how emotions and presentation influence rational decision-making.",
        ],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THINKING = "<|thinking|>{text}<|/thinking|>"
PAR = "<|par|>{text}<|/par|>"
TITLE = "<|title|>{text}<|/title|>"
HEADING = "<|heading|>{text}<|/heading|>"
DIVIDER = "<|divider|>"
ENDTEXT = "<|endoftext|>"


def format_steps(title: str, steps: list[str], rng: random.Random) -> str:
    lines = [TITLE.format(text=title), ""]
    add_final_marker = rng.random() < 0.2  # small chance of heading for final

    for i, step in enumerate(steps):
        tphrase = rng.choice(THINKING_PHRASES)
        lines.append(THINKING.format(text=tphrase))
        lines.append("")
        lines.append(PAR.format(text=step))
        lines.append("")

    lines.append(ENDTEXT)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic thinking-trace training data.")
    ap.add_argument("--count", type=int, default=1000, help="Number of files to generate")
    ap.add_argument("--out", default=OUT_DIR, help="Output directory")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    existing = [f for f in os.listdir(args.out) if f.startswith("synthetic") and f.endswith(".txt")]
    if existing:
        # Find the highest existing number
        nums = []
        for f in existing:
            m = re.match(r"synthetic(\d+)\.txt", f)
            if m:
                nums.append(int(m.group(1)))
        start = max(nums) + 1 if nums else 1
    else:
        start = 1

    for i in range(args.count):
        topic = rng.choice(TOPICS)
        idx = start + i
        content = format_steps(topic["title"], topic["steps"], rng)
        fname = f"synthetic{idx:04d}.txt"
        path = os.path.join(args.out, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {args.count} files -> {args.out}")


if __name__ == "__main__":
    main()
