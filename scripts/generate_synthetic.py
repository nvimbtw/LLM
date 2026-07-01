#!/usr/bin/env python3
"""
Generate synthetic thinking-trace training data with extended inner monologue.

Each file contains <|thinking|>...</|thinking|> blocks (long-form reasoning where
the model talks itself through the problem) interleaved with <|par|>...</|par|>
response blocks, wrapped in project markup tokens.

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


TOPICS = [
    # ---- MATH ----
    {
        "title": "Solving a Quadratic by Factoring",
        "steps": [
            ("Solve x^2 - 5x + 6 = 0 by factoring.",
             "We need two numbers that multiply to 6 and add to -5. The pair (-2, -3) works because (-2)*(-3) = 6 and (-2)+(-3) = -5."),
            ("So x^2 - 5x + 6 = (x - 2)(x - 3) = 0. Setting each factor to zero: x - 2 = 0 or x - 3 = 0.",
             "The solutions are x = 2 and x = 3."),
        ],
    },
    {
        "title": "Completing the Square",
        "steps": [
            ("Solve x^2 + 6x - 7 = 0 by completing the square.",
             "Move the constant: x^2 + 6x = 7. Take half of 6, square it: (6/2)^2 = 9. Add to both sides: x^2 + 6x + 9 = 16."),
            ("The left side is (x + 3)^2 = 16. Take square root: x + 3 = ±4.",
             "x = -3 + 4 = 1 or x = -3 - 4 = -7."),
        ],
    },
    {
        "title": "Quadratic Formula",
        "steps": [
            ("Solve 2x^2 - 4x - 3 = 0 using the quadratic formula.",
             "Quadratic formula: x = [-b ± sqrt(b^2 - 4ac)] / (2a). Here a = 2, b = -4, c = -3. The discriminant: b^2 - 4ac = 16 - 4(2)(-3) = 16 + 24 = 40."),
            ("sqrt(40) = 2*sqrt(10). x = [4 ± 2*sqrt(10)] / 4 = [2 ± sqrt(10)] / 2.",
             "The two solutions are (2 + sqrt(10))/2 and (2 - sqrt(10))/2."),
        ],
    },
    {
        "title": "Systems of Linear Equations",
        "steps": [
            ("Solve: 3x + 2y = 12 and 5x - 2y = 4.",
             "Adding eliminates y: (3x+2y)+(5x-2y) = 12+4 gives 8x = 16, so x = 2. Substitute back: 3(2)+2y = 12 -> 2y = 6 -> y = 3."),
            ("The solution is x = 2, y = 3.",
             "Checking: 5(2)-2(3)=10-6=4. Both equations satisfied."),
        ],
    },
    {
        "title": "Distance Rate Time Problem",
        "steps": [
            ("A train leaves Station A at 60 mph. Two hours later, a second train leaves Station A at 80 mph on the same track. How long will it take the second train to catch the first?",
             "Let t be the time after the second train departs. The first train travels for t+2 hours at 60 mph, covering 60(t+2) miles. The second train travels t hours at 80 mph, covering 80t miles."),
            ("When the second catches the first, distances are equal: 60(t+2) = 80t -> 60t + 120 = 80t -> 120 = 20t -> t = 6.",
             "The second train catches the first after 6 hours. That is 8 hours after the first departed. Checking: first goes 60*8 = 480 miles, second goes 80*6 = 480 miles. Matches."),
        ],
    },
    {
        "title": "Work Problem",
        "steps": [
            ("Pipe A fills a tank in 4 hours. Pipe B fills the same tank in 6 hours. How long together?",
             "Pipe A rate = 1/4 tank per hour. Pipe B rate = 1/6 tank per hour. Combined = 1/4 + 1/6 = 3/12 + 2/12 = 5/12 tank per hour."),
            ("Time = 1 / (5/12) = 12/5 = 2.4 hours.",
             "That is 2 hours and 24 minutes. Makes sense — should be less than either alone but more than half of the faster pipe."),
        ],
    },
    {
        "title": "Mixture Problem",
        "steps": [
            ("How many liters of 20% saline must be mixed with 30 L of 50% saline to get 40% solution?",
             "Let x = liters of 20% solution. Salt from 20%: 0.2x. Salt from 50%: 0.5 * 30 = 15. Total volume: x + 30."),
            ("(0.2x + 15) / (x + 30) = 0.4 -> 0.2x + 15 = 0.4x + 12 -> 3 = 0.2x -> x = 15.",
             "15 liters of 20% saline. Quick check: total salt = 0.2(15)+15 = 18, total volume = 45, 18/45 = 0.4. Correct."),
        ],
    },
    {
        "title": "Area of a Circle",
        "steps": [
            ("A circular garden has radius 7 meters. What is area? (Use pi = 22/7)",
             "Area = pi * r^2 = (22/7) * 7^2 = (22/7) * 49. 49/7 = 7, so A = 22 * 7."),
            ("A = 154 square meters.",
             "That is about 154 m^2. Could also approximate with 3.14: 3.14 * 49 = 153.86. Close enough."),
        ],
    },
    {
        "title": "Pythagorean Theorem",
        "steps": [
            ("A right triangle has legs 5 and 12. Find the hypotenuse.",
             "Pythagorean theorem: a^2 + b^2 = c^2. Here 5^2 + 12^2 = 25 + 144 = 169."),
            ("c = sqrt(169) = 13.",
             "13 is a nice integer. A 5-12-13 triangle is a well-known Pythagorean triple."),
        ],
    },
    {
        "title": "Volume of a Sphere",
        "steps": [
            ("What is the volume of a sphere with radius 6 cm? (Use pi = 3.14)",
             "V = (4/3) * pi * r^3 = (4/3) * 3.14 * 216."),
            ("216 * 3.14 = 678.24. (4/3) * 678.24 = 904.32.",
             "Approximately 904.32 cubic centimeters. That seems reasonable for a sphere of diameter 12 cm."),
        ],
    },
    {
        "title": "Slope of a Line",
        "steps": [
            ("Find the slope through (3, 4) and (7, 10).",
             "Slope = (y2 - y1) / (x2 - x1) = (10 - 4) / (7 - 3)."),
            ("6/4 simplifies to 3/2.",
             "Slope is 1.5. That means for every 2 units right, the line goes up 3 units."),
        ],
    },
    {
        "title": "Arithmetic Sequence Sum",
        "steps": [
            ("Sum the sequence 3, 7, 11, 15, ... up to 20 terms.",
             "First term a = 3, common difference d = 4, n = 20. The 20th term = a + (n-1)d = 3 + 19*4 = 79."),
            ("Sum = n * (first + last) / 2 = 20 * (3 + 79) / 2 = 20 * 82 / 2 = 20 * 41 = 820.",
             "So the sum of the first 20 terms is 820. That formula works for any arithmetic sequence."),
        ],
    },
    {
        "title": "Geometric Sequence",
        "steps": [
            ("Find the 8th term of: 2, 6, 18, 54, ...",
             "First term a = 2, common ratio r = 3. The nth term is a * r^(n-1)."),
            ("8th term: 2 * 3^7 = 2 * 2187 = 4374.",
             "That grows fast. 3^7 is 2187, doubled to 4374. The sequence 2,6,18,54,162,486,1458,4374 checks out."),
        ],
    },
    {
        "title": "Exponential Growth",
        "steps": [
            ("Bacteria double every 3 hours. Start with 100. How many after 24 hours?",
             "Growth: N = N0 * 2^(t/d). N0 = 100, t = 24, d = 3. N = 100 * 2^(24/3) = 100 * 2^8."),
            ("2^8 = 256. N = 100 * 256 = 25,600.",
             "From 100 to 25,600 in 24 hours. That is exponential growth for you — 8 doublings."),
        ],
    },
    {
        "title": "Logarithm Evaluation",
        "steps": [
            ("Evaluate log_3(81) without a calculator.",
             "We need x such that 3^x = 81. Powers of 3: 3^1 = 3, 3^2 = 9, 3^3 = 27, 3^4 = 81."),
            ("Since 3^4 = 81, log_3(81) = 4.",
             "Straightforward. The logarithm is just the exponent needed."),
        ],
    },
    {
        "title": "Factorial Simplification",
        "steps": [
            ("Simplify: 10! / 7!",
             "10! = 10 * 9 * 8 * 7!. So 10! / 7! = 10 * 9 * 8."),
            ("10 * 9 * 8 = 720.",
             "Much simpler than expanding both factorials fully."),
        ],
    },
    {
        "title": "Combinations",
        "steps": [
            ("How many ways to choose a committee of 3 from 10 people?",
             "C(10,3) = 10! / (3! * 7!) = (10 * 9 * 8) / (3 * 2 * 1) = 720 / 6."),
            ("720 / 6 = 120 possible committees.",
             "That makes sense — a lot of ways to pick from 10 people."),
        ],
    },
    {
        "title": "Probability of Independent Events",
        "steps": [
            ("What is P(roll a 6 AND flip heads)?",
             "P(6) = 1/6. P(heads) = 1/2. Independent events multiply."),
            ("P(both) = 1/6 * 1/2 = 1/12.",
             "About 8.3%. Intuitively, half the time you flip heads and only a sixth of those are also a 6."),
        ],
    },
    {
        "title": "Expected Value",
        "steps": [
            ("Game pays $10 for rolling a 6, $5 for a 5, nothing otherwise. Find expected value.",
             "P(6) = 1/6, P(5) = 1/6, P(other) = 4/6. E = (1/6)*10 + (1/6)*5 + (4/6)*0."),
            ("E = 10/6 + 5/6 = 15/6 = $2.50.",
             "So on average you win $2.50 per roll. Over many rolls, that is what you would expect."),
        ],
    },
    {
        "title": "Standard Deviation",
        "steps": [
            ("Find standard deviation of: 2, 4, 4, 4, 5, 5, 7, 9.",
             "Mean = 40/8 = 5. Deviations: -3, -1, -1, -1, 0, 0, 2, 4. Squared: 9, 1, 1, 1, 0, 0, 4, 16."),
            ("Variance = 32/8 = 4. Standard deviation = sqrt(4) = 2.",
             "So on average, values are about 2 units away from the mean of 5."),
        ],
    },
    # ---- PHYSICS ----
    {
        "title": "Kinematic Equation",
        "steps": [
            ("Car accelerates from rest at 2 m/s^2 for 5 s. How far?",
             "s = ut + (1/2)at^2. u = 0, a = 2, t = 5. s = 0 + 0.5 * 2 * 25."),
            ("s = 25 meters.",
             "From rest, covering 25 meters in 5 seconds with that acceleration feels about right."),
        ],
    },
    {
        "title": "Free Fall",
        "steps": [
            ("Ball dropped from 45 m. Time to hit ground? (g = 10 m/s^2)",
             "s = (1/2)gt^2. 45 = 5t^2. t^2 = 9."),
            ("t = 3 seconds.",
             "Three seconds of free fall from 45 meters. That sounds realistic."),
        ],
    },
    {
        "title": "Newton's Second Law",
        "steps": [
            ("A 20 N force acts on a 5 kg mass. Find acceleration.",
             "F = ma -> a = F/m = 20 / 5."),
            ("a = 4 m/s^2.",
             "The heavier the mass, the smaller the acceleration for the same force."),
        ],
    },
    {
        "title": "Friction Force",
        "steps": [
            ("10 kg box pulled across floor, mu_k = 0.3. Find friction force. (g = 9.8 m/s^2)",
             "Normal force N = mg = 10 * 9.8 = 98 N. Friction f = mu_k * N = 0.3 * 98."),
            ("f = 29.4 N.",
             "That is the force opposing the motion. You would need to pull with more than 29.4 N to keep it moving at constant speed."),
        ],
    },
    {
        "title": "Work Done",
        "steps": [
            ("Lift a 15 kg object 2 meters. Work done? (g = 9.8)",
             "Work = force * distance. Force = mg = 15 * 9.8 = 147 N. Work = 147 * 2."),
            ("Work = 294 J.",
             "That is about the energy needed to lift a decently heavy box onto a shelf."),
        ],
    },
    {
        "title": "Kinetic Energy",
        "steps": [
            ("1000 kg car at 20 m/s. Kinetic energy?",
             "KE = 0.5 * m * v^2 = 0.5 * 1000 * 20^2 = 500 * 400."),
            ("KE = 200,000 J = 200 kJ.",
             "That is a significant amount of energy. Explains why car crashes are so destructive."),
        ],
    },
    {
        "title": "Potential Energy",
        "steps": [
            ("2 kg book on a 3 m shelf. Potential energy? (g = 9.8)",
             "PE = mgh = 2 * 9.8 * 3."),
            ("PE = 58.8 J.",
             "Fifty-eight point eight joules. If it fell, that energy would convert to kinetic."),
        ],
    },
    {
        "title": "Power Calculation",
        "steps": [
            ("A motor does 5000 J of work in 10 s. Power?",
             "P = W / t = 5000 / 10."),
            ("P = 500 W.",
             "Five hundred watts. That is about what a small appliance draws."),
        ],
    },
    {
        "title": "Ohm's Law",
        "steps": [
            ("12 V battery across 4 ohm resistor. Current?",
             "I = V / R = 12 / 4."),
            ("I = 3 A.",
             "Three amperes. If this were a higher resistance, current would drop proportionally."),
        ],
    },
    {
        "title": "Power in a Circuit",
        "steps": [
            ("Device draws 2 A at 120 V. Power consumed?",
             "P = VI = 120 * 2."),
            ("P = 240 W.",
             "240 watts. That is typical for a desktop computer power supply."),
        ],
    },
    {
        "title": "Resistors in Series",
        "steps": [
            ("Three resistors: 2, 3, and 5 ohms in series. Total resistance?",
             "R_total = R1 + R2 + R3 = 2 + 3 + 5."),
            ("R_total = 10 ohms.",
             "In series, resistance adds directly. The current has to go through each one sequentially."),
        ],
    },
    {
        "title": "Resistors in Parallel",
        "steps": [
            ("6 ohm and 3 ohm resistors in parallel. Total resistance?",
             "1/R_total = 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2."),
            ("R_total = 2 ohms.",
             "Lower than either individual resistor. More paths means less overall resistance."),
        ],
    },
    {
        "title": "Wavelength and Frequency",
        "steps": [
            ("Wave has frequency 500 Hz and speed 340 m/s. Wavelength?",
             "v = f * lambda -> lambda = v / f = 340 / 500."),
            ("lambda = 0.68 m.",
             "A sound wave with 500 Hz has wavelength about 0.68 meters in air."),
        ],
    },
    {
        "title": "Snell's Law",
        "steps": [
            ("Light from air (n=1) into water (n=1.33) at 30 degree incidence. Refraction angle?",
             "n1*sin(theta1) = n2*sin(theta2). 1*sin(30) = 1.33*sin(theta2). sin(30)=0.5."),
            ("0.5 = 1.33*sin(theta2). sin(theta2) = 0.376. theta2 = arcsin(0.376) ≈ 22.1 degrees.",
             "The light bends toward the normal as it enters the denser medium. 22.1 degrees makes sense."),
        ],
    },
    {
        "title": "Ideal Gas Law",
        "steps": [
            ("2 moles of gas at 27 C and 1 atm. Volume? (R = 0.0821)",
             "PV = nRT. T = 27 + 273 = 300 K. V = nRT/P = 2 * 0.0821 * 300 / 1."),
            ("V = 49.26 L.",
             "About 49 liters for 2 moles at room temperature and atmospheric pressure."),
        ],
    },
    {
        "title": "Heat Transfer",
        "steps": [
            ("Heat to raise 500 g water from 20 C to 80 C? (c = 4.18 J/g*C)",
             "Q = mc*delta_T = 500 * 4.18 * (80 - 20) = 500 * 4.18 * 60."),
            ("Q = 125,400 J = 125.4 kJ.",
             "That is a fair amount of heat. Water has a high specific heat capacity."),
        ],
    },
    {
        "title": "Momentum Conservation",
        "steps": [
            ("2 kg ball at 4 m/s hits stationary 1 kg ball. They stick together. Final velocity?",
             "m1*v1 + m2*v2 = (m1+m2)*v_final. 2*4 + 1*0 = (2+1)*v_final = 8."),
            ("v_final = 8/3 ≈ 2.67 m/s.",
             "The combined mass moves slower than the original ball, conserving momentum."),
        ],
    },
    {
        "title": "Centripetal Force",
        "steps": [
            ("0.5 kg mass swings in radius 2 m at 3 m/s. Centripetal force?",
             "F = mv^2/r = 0.5 * 3^2 / 2 = 0.5 * 9 / 2."),
            ("F = 2.25 N.",
             "Not a huge force. If the speed doubled, the force would quadruple since it depends on v^2."),
        ],
    },
    {
        "title": "Buoyant Force",
        "steps": [
            ("0.1 m^3 object submerged in water (density 1000 kg/m^3). Buoyant force? (g = 9.8)",
             "F_b = rho * V * g = 1000 * 0.1 * 9.8."),
            ("F_b = 980 N.",
             "That is the weight of the displaced water. Equivalent to about 100 kg of water."),
        ],
    },
    {
        "title": "Simple Pendulum Period",
        "steps": [
            ("Period of a 1 m pendulum? (g = 9.8, pi = 3.14)",
             "T = 2*pi * sqrt(L/g) = 6.28 * sqrt(1/9.8). sqrt(1/9.8) = sqrt(0.102) = 0.319."),
            ("T = 6.28 * 0.319 = 2.00 seconds.",
             "A 1-meter pendulum takes about 2 seconds per swing. That is why pendulum clocks work so well."),
        ],
    },
    # ---- PROGRAMMING / ALGORITHMS ----
    {
        "title": "Finding the Maximum in an Array",
        "steps": [
            ("Find maximum in [3, 7, 2, 9, 5] using linear scan.",
             "Initialize max = arr[0] = 3. Compare: 7 > 3 -> max = 7. 2 < 7 -> skip. 9 > 7 -> max = 9. 5 < 9 -> skip."),
            ("Maximum is 9. This is O(n) time, O(1) space — optimal for an unsorted array.",
             "You cannot do better than O(n) because you have to look at every element."),
        ],
    },
    {
        "title": "Binary Search",
        "steps": [
            ("Find 15 in sorted [2, 5, 8, 12, 15, 19, 22].",
             "Low=0, high=6, mid=3. arr[3]=12 < 15, search right. Low=4, high=6, mid=5. arr[5]=19 > 15, search left."),
            ("Low=4, high=4, mid=4. arr[4]=15. Found at index 4 in 3 comparisons.",
             "Logarithmic search is extremely efficient for sorted data."),
        ],
    },
    {
        "title": "Bubble Sort Trace",
        "steps": [
            ("Trace one pass of bubble sort on [5, 3, 8, 1, 2].",
             "Compare 5&3: swap -> [3,5,8,1,2]. 5<8: no swap. 8>1: swap -> [3,5,1,8,2]. 8>2: swap -> [3,5,1,2,8]."),
            ("After one pass, the largest element (8) has bubbled to the end. Repeat for n-1 elements.",
             "Bubble sort is O(n^2) average and worst case. Simple but slow for large data."),
        ],
    },
    {
        "title": "Checking for Palindrome",
        "steps": [
            ("Is 'racecar' a palindrome?",
             "Two pointers: left at 0, right at 6. Compare: r==r, advance. a==a, c==c, e==e. All matched."),
            ("'racecar' is a palindrome. Both forward and backward read the same.",
             "This two-pointer approach is O(n) time and O(1) space."),
        ],
    },
    {
        "title": "Fibonacci (Recursive)",
        "steps": [
            ("Compute the 6th Fibonacci number recursively. F(1)=F(2)=1.",
             "F(6) = F(5)+F(4). F(5) = F(4)+F(3). F(4) = F(3)+F(2). F(3) = F(2)+F(1) = 1+1 = 2."),
            ("F(4) = 2+1=3. F(5) = 3+2=5. F(6) = 5+3=8.",
             "Recursion is elegant but exponential. For larger n, DP is much better."),
        ],
    },
    {
        "title": "Factorial (Iterative)",
        "steps": [
            ("Compute 5! iteratively.",
             "Start result=1. *2 -> 2. *3 -> 6. *4 -> 24. *5 -> 120."),
            ("5! = 120.",
             "The iterative approach uses constant stack space unlike recursion."),
        ],
    },
    {
        "title": "String Reversal",
        "steps": [
            ("Reverse 'hello' in place.",
             "Swap h<->o: 'oellh'. Swap e<->l: 'olleh'. Middle character stays."),
            ("'hello' reversed is 'olleh'. Two-pointer swap in O(n) time.",
             "No extra space needed beyond a temporary variable for the swap."),
        ],
    },
    {
        "title": "Finding Duplicates",
        "steps": [
            ("Find duplicates in [1, 3, 4, 2, 3, 1, 5].",
             "Use a hash set: insert 1, 3, 4, 2. When inserting 3 again, it is already present. Insert 1 again, also present."),
            ("Duplicates found: 3 and 1. O(n) time, O(n) space.",
             "If we needed O(1) space and could modify the array, we could use the values as indices with negation."),
        ],
    },
    {
        "title": "Linked List Reversal",
        "steps": [
            ("Reverse 1 -> 2 -> 3 -> null.",
             "prev=null, curr=1. Save next=2, set curr.next=null. prev=1, curr=2. Save next=3, set curr.next=1."),
            ("prev=2, curr=3. Save next=null, set curr.next=2. prev=3. New head is 3. Reversed: 3->2->1->null.",
             "Reversing a linked list iteratively requires three pointers."),
        ],
    },
    {
        "title": "Depth-First Search",
        "steps": [
            ("DFS on {A: [B,C], B: [D], C: [E], D: [], E: []} starting at A.",
             "Visit A. Recursively visit B. Visit B. Recursively visit D. Visit D. Backtrack to B. Backtrack to A."),
            ("Visit C. Visit E. DFS order: A, B, D, C, E.",
             "DFS goes deep before wide, using a stack (explicit or recursion)."),
        ],
    },
    {
        "title": "Breadth-First Search",
        "steps": [
            ("BFS on {A: [B,C], B: [D], C: [E], D: [], E: []} starting at A.",
             "Enqueue A. Dequeue A, visit A. Enqueue B,C. Dequeue B, visit B. Enqueue D. Dequeue C, visit C. Enqueue E."),
            ("Dequeue D, visit D. Dequeue E, visit E. BFS order: A, B, C, D, E.",
             "BFS goes wide before deep, using a queue."),
        ],
    },
    {
        "title": "Two Sum Problem",
        "steps": [
            ("Given [2,7,11,15] and target 9, find two numbers summing to 9.",
             "Use hash map. For 2, check if 7 (9-2) is in map. Not yet, add 2->0. For 7, check if 2 is in map. Yes at index 0."),
            ("Indices [0,1] with values 2 + 7 = 9. O(n) time, O(n) space.",
             "The brute force approach would be O(n^2). The hash map trades space for speed."),
        ],
    },
    {
        "title": "FizzBuzz",
        "steps": [
            ("Print numbers 1 to 15 with FizzBuzz rules: Fizz for multiples of 3, Buzz for 5, FizzBuzz for both.",
             "1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz. Classic interview screening problem. Tests basic control flow understanding."),
        ],
    },
    {
        "title": "Time Complexity Analysis",
        "steps": [
            ("What is the complexity of nested loops: for i in range(n): for j in range(n): print(i*j)?",
             "Outer loop runs n times. For each outer iteration, inner loop runs n times. Total: n * n = n^2 operations."),
            ("Time complexity: O(n^2). If n doubles, operations quadruple.",
             "Quadratic complexity becomes slow quickly as n grows."),
        ],
    },
    {
        "title": "Prime Number Check",
        "steps": [
            ("Is 17 a prime number?",
             "Check divisibility up to sqrt(17) ≈ 4. 17 mod 2 = 1, mod 3 = 2, mod 4 = 1. No divisors found."),
            ("17 is a prime number. It has no positive divisors other than 1 and itself.",
             "Checking up to sqrt(n) is sufficient because factors come in pairs."),
        ],
    },
    {
        "title": "Number of Digits",
        "steps": [
            ("How many digits in 12345?",
             "Repeatedly divide by 10: 12345/10=1234 (1), 1234/10=123 (2), 123/10=12 (3), 12/10=1 (4), 1/10=0 (5)."),
            ("12345 has 5 digits. Alternatively: floor(log10(12345)) + 1 = 4 + 1 = 5.",
             "The log approach works for any positive integer."),
        ],
    },
    {
        "title": "Fibonacci (Dynamic Programming)",
        "steps": [
            ("Compute F(50) efficiently.",
             "Naive recursion is exponential O(2^n). Instead, use bottom-up DP: F(1)=1, F(2)=1. For i=3..50: F(i) = F(i-1)+F(i-2)."),
            ("F(50) = 12,586,269,025. O(n) time, O(1) space (keep only last two values).",
             "DP transforms an exponential problem into a linear one."),
        ],
    },
    # ---- LOGIC PUZZLES ----
    {
        "title": "River Crossing Puzzle",
        "steps": [
            ("Farmer must cross with wolf, goat, and cabbage. Boat holds him plus one item. Wolf eats goat if left alone; goat eats cabbage.",
             "Cannot leave wolf+goat or goat+cabbage unattended. Take goat first. Leave goat on far side, return alone."),
            ("Take wolf across. Bring goat back. Leave goat, take cabbage across. Return alone. Take goat across.",
             "The key insight is bringing the goat back on the second trip to break the conflict pairs."),
        ],
    },
    {
        "title": "The Missing Dollar Riddle",
        "steps": [
            ("Three friends pay $30. Room is $25. Bellboy keeps $2, returns $1 each. Friends paid $9 each = $27, plus bellboy's $2 = $29. Where is the missing dollar?",
             "The $27 already includes the $2 the bellboy kept ($25 room + $2 kept = $27). Adding the $2 again double-counts it."),
            ("$27 + $3 returned = $30. No missing dollar. The puzzle is a misdirection in the arithmetic.",
             "This is a classic example of how mislabeling operations creates a false paradox."),
        ],
    },
    {
        "title": "The Two Doors Riddle",
        "steps": [
            ("Two doors: one treasure, one death. Two guards: one always lies, one always tells truth. You can ask one question.",
             "Asking 'which door has treasure?' gives two different answers and you don't know who is who."),
            ("Ask either: 'What would the other guard say is the treasure door?' Choose the opposite. Both guards point to the wrong door.",
             "This works because liars lie about what truth-tellers would say, and truth-tellers report the liar's false answer."),
        ],
    },
    {
        "title": "The 12 Coin Problem",
        "steps": [
            ("12 coins, one counterfeit (different weight). Find it with 3 weighings on a balance scale.",
             "Weigh 4 vs 4. If balanced, fake is among remaining 4 — use 2 weighings to find it. If unbalanced, fake among those 8 with known heavy/light markings."),
            ("With careful planning, the third weighing always identifies the fake and whether it is heavier or lighter.",
             "The key is extracting the maximum information from each weighing by the way coins are grouped."),
        ],
    },
    {
        "title": "Blue Eyes Puzzle",
        "steps": [
            ("100 blue-eyed and 100 brown-eyed people. Guru says 'At least one has blue eyes.' Everyone sees others' eye colors but not their own. No communication. What happens?",
             "Solved by induction. If 1 person had blue eyes, they would see none, deduce they are that one, and leave the first night."),
            ("If 2 have blue eyes, each sees 1 other. If that other doesn't leave night 1, both know they are blue and leave night 2. By induction, all 100 blue-eyed leave on night 100.",
             "The guru's statement provides common knowledge, which is different from everyone knowing a fact individually."),
        ],
    },
    {
        "title": "Birthday Paradox",
        "steps": [
            ("Probability that 23 people share a birthday?",
             "Compute complement: P(all different) = (365/365)*(364/365)*...*(343/365) ≈ 0.4927."),
            ("P(at least one match) = 1 - 0.4927 = 0.5073 ≈ 50.7%.",
             "Only 23 people needed for better than even odds. Intuitively surprising, but mathematically sound."),
        ],
    },
    {
        "title": "The Monty Hall Problem",
        "steps": [
            ("3 doors: 1 car, 2 goats. Pick door 1. Host opens door 3 revealing a goat. Should you switch?",
             "Initially P(car behind your door) = 1/3. P(car behind other two) = 2/3. Host always opens a goat door, concentrating the 2/3 on the remaining door."),
            ("Switching gives 2/3 chance. Staying gives 1/3. Always switch.",
             "This is counterintuitive because the host's knowledge changes the probabilities."),
        ],
    },
    {
        "title": "The Chessboard and Rice",
        "steps": [
            ("1 grain on first square, double each time. Total on 64 squares?",
             "Sum = 2^0 + 2^1 + ... + 2^63 = 2^64 - 1."),
            ("2^64 = 18,446,744,073,709,551,616. Total = 18,446,744,073,709,551,615 grains ≈ 18.4 quintillion.",
             "More rice than has ever been produced in human history. Exponential growth is deceptive."),
        ],
    },
    {
        "title": "Handshake Problem",
        "steps": [
            ("10 people each shake hands once with everyone else. Total handshakes?",
             "Each of 10 people shakes 9 hands = 90. But that counts each handshake twice (A-B and B-A)."),
            ("90 / 2 = 45 handshakes. Formula: n(n-1)/2 = 10*9/2 = 45.",
             "This is the same as the number of pairs or edges in a complete graph."),
        ],
    },
    {
        "title": "The Water Jug Problem",
        "steps": [
            ("Measure exactly 4 gallons using a 5-gallon and 3-gallon jug.",
             "Fill 5. Pour 5 into 3 until 3 is full, leaving 2 in 5. Empty 3. Pour the 2 from 5 into 3. Fill 5."),
            ("Pour from 5 into 3 until 3 is full (needs 1). That leaves exactly 4 in 5.",
             "This works because the difference between the jug sizes is 2, and we use that strategically."),
        ],
    },
    {
        "title": "Clock Angle Problem",
        "steps": [
            ("Angle between hour and minute hands at 3:15?",
             "At 3:00, hour hand at 90 degrees, minute at 0. In 15 minutes, minute moves 90 degrees (15/60*360). Hour moves 7.5 degrees (15/60*30)."),
            ("Hour hand at 90+7.5=97.5. Minute at 90. Angle = |97.5-90| = 7.5 degrees.",
             "The hour hand keeps moving too, not just jumping between hours."),
        ],
    },
    {
        "title": "The Race Track Problem",
        "steps": [
            ("Two laps of a 400 m track: first at 4 m/s, second at 6 m/s. Average speed?",
             "Time for lap 1: 400/4 = 100 s. Lap 2: 400/6 = 66.67 s. Total: 800 m in 166.67 s."),
            ("Average speed = 800/166.67 = 4.8 m/s. Not the arithmetic mean of 5 m/s because more time is spent at the slower speed.",
             "Harmonic mean would give the correct answer directly: 2/(1/4+1/6) = 4.8."),
        ],
    },
    {
        "title": "The Poisoned Wine",
        "steps": [
            ("1000 bottles of wine, one poisoned. 10 test subjects. Find poisoned bottle in 24 hours.",
             "Number bottles 0-999 in binary (10 bits). Subject k drinks from bottles where bit k is 1."),
            ("After 24 hours, pattern of dead subjects forms the binary number of the poisoned bottle. 2^10 = 1024 ≥ 1000.",
             "Each subject tests half the bottles. The binary encoding is maximally efficient."),
        ],
    },
    # ---- CHEMISTRY ----
    {
        "title": "Balancing Chemical Equations",
        "steps": [
            ("Balance: H2 + O2 -> H2O.",
             "Left: 2H, 2O. Right: 2H, 1O. Put coefficient 2 before H2O: H2 + O2 -> 2H2O. Now right has 4H, 2O. Put 2 before H2."),
            ("2H2 + O2 -> 2H2O. Balanced: 4H and 2O on each side.",
             "Always balance oxygen last in hydrocarbon reactions. Start with the most complex molecule."),
        ],
    },
    {
        "title": "Mole Calculation",
        "steps": [
            ("How many moles in 36 g of water? (H=1, O=16)",
             "Molar mass of H2O = 2*1 + 16 = 18 g/mol. Moles = mass / molar mass = 36 / 18."),
            ("2 moles of water. That is 2 * 6.022*10^23 = 1.204*10^24 molecules.",
             "The mole is a count of particles, not a measure of mass directly."),
        ],
    },
    {
        "title": "pH Calculation",
        "steps": [
            ("pH of 0.001 M HCl?",
             "HCl is a strong acid, dissociates completely. [H+] = 0.001 M = 10^(-3) M."),
            ("pH = -log10(10^(-3)) = 3. This is a moderately acidic solution.",
             "Each unit of pH represents a tenfold change in H+ concentration."),
        ],
    },
    {
        "title": "Molarity Calculation",
        "steps": [
            ("Molarity of 58.5 g NaCl in 500 mL of water? (Na=23, Cl=35.5)",
             "Molar mass NaCl = 23 + 35.5 = 58.5 g/mol. Moles = 58.5/58.5 = 1 mol."),
            ("Molarity = 1 mol / 0.5 L = 2 M.",
             "Concentration is moles per liter of solution, not solvent."),
        ],
    },
    {
        "title": "Stoichiometry",
        "steps": [
            ("Grams of CO2 from 24 g CH4? CH4 + 2O2 -> CO2 + 2H2O. (C=12, H=1, O=16)",
             "Molar mass CH4 = 12+4 = 16 g/mol. Moles CH4 = 24/16 = 1.5. 1:1 mole ratio with CO2. Molar mass CO2 = 12+32 = 44 g/mol."),
            ("Mass CO2 = 1.5 * 44 = 66 grams.",
             "The balanced equation gives the mole ratios needed for this calculation."),
        ],
    },
    {
        "title": "Oxidation States",
        "steps": [
            ("Oxidation state of Mn in KMnO4?",
             "K: +1, O: -2 each (x4 = -8). The compound is neutral: +1 + Mn + (-8) = 0."),
            ("Mn = +7. Permanganate has manganese in its highest common oxidation state.",
             "Oxidation states are bookkeeping tools; they don't always reflect actual charge distribution."),
        ],
    },
    {
        "title": "Electronegativity Trends",
        "steps": [
            ("Arrange F, Cl, Br, I by increasing electronegativity.",
             "Electronegativity decreases down group 17 as atomic radius increases and valence electrons are further from the nucleus."),
            ("I < Br < Cl < F. Fluorine is the most electronegative element at 3.98 on the Pauling scale.",
             "Electronegativity also increases left to right across a period."),
        ],
    },
    {
        "title": "Gas Stoichiometry",
        "steps": [
            ("Volume of O2 at STP needed to react with 10 L of H2? 2H2 + O2 -> 2H2O.",
             "At STP, equal volumes contain equal moles (Avogadro's law). The ratio H2:O2 = 2:1 by volume."),
            ("10 L H2 requires 5 L O2 at STP.",
             "Gas volumes at the same T and P are directly proportional to moles."),
        ],
    },
    {
        "title": "Enthalpy Change",
        "steps": [
            ("Enthalpy change when 2 moles H2O form? H2 + 1/2 O2 -> H2O, delta_H = -286 kJ/mol.",
             "The given delta_H is per mole of H2O formed. For 2 moles: 2 * (-286 kJ)."),
            ("delta_H = -572 kJ. The reaction is exothermic, releasing 572 kJ of heat.",
             "The negative sign tells us heat is released to the surroundings."),
        ],
    },
    {
        "title": "Half-Life Calculation",
        "steps": [
            ("100 g of isotope with half-life 10 years. Remaining after 40 years?",
             "After each half-life, mass halves. 40 years is 4 half-lives."),
            ("100 -> 50 -> 25 -> 12.5 -> 6.25 g. Formula: 100 * (1/2)^(40/10) = 100 * 1/16 = 6.25 g.",
             "Exponential decay follows a predictable pattern regardless of starting amount."),
        ],
    },
    {
        "title": "Limiting Reactant",
        "steps": [
            ("28 g N2 reacts with 6 g H2. Limiting reactant? N2 + 3H2 -> 2NH3. (N=14, H=1)",
             "Moles N2 = 28/28 = 1. Moles H2 = 6/2 = 3. Balanced equation needs N2:H2 = 1:3. Exactly stoichiometric."),
            ("Neither is limiting. Both are fully consumed. Theoretical yield of NH3 = 2 * 17 = 34 g.",
             "When reactants are in exact proportion, there is no limiting reactant."),
        ],
    },
    {
        "title": "Percent Yield",
        "steps": [
            ("Theoretical yield 50 g, actual 40 g. Percent yield?",
             "Percent yield = (actual / theoretical) * 100% = (40/50) * 100%."),
            ("Percent yield = 80%. That is a reasonable yield for many reactions.",
             "Losses come from incomplete reactions, side reactions, and purification."),
        ],
    },
    # ---- BIOLOGY ----
    {
        "title": "Mitosis Phases",
        "steps": [
            ("List the phases of mitosis in order and what happens in each.",
             "Prophase: chromosomes condense, nuclear envelope breaks down. Metaphase: chromosomes align at the metaphase plate. Anaphase: sister chromatids separate. Telophase: nuclear envelopes reform."),
            ("Cytokinesis follows, dividing the cytoplasm. The result is two genetically identical daughter cells.",
             "Mitosis is the basis for growth and repair in multicellular organisms."),
        ],
    },
    {
        "title": "DNA Base Pairing",
        "steps": [
            ("Complementary strand for A-T-G-C-C-A?",
             "DNA base pairing: A pairs with T, G pairs with C. Direction matters: strands are antiparallel."),
            ("Complement: T-A-C-G-G-T.",
             "This specific pairing is due to hydrogen bonding patterns between the bases."),
        ],
    },
    {
        "title": "Cellular Respiration Summary",
        "steps": [
            ("Overall equation for aerobic cellular respiration?",
             "C6H12O6 + 6O2 -> 6CO2 + 6H2O + 36-38 ATP. Glucose is oxidized, oxygen is reduced."),
            ("Three stages: glycolysis (cytoplasm), Krebs cycle (mitochondrial matrix), and the electron transport chain (inner mitochondrial membrane).",
             "Most ATP comes from the electron transport chain via oxidative phosphorylation."),
        ],
    },
    {
        "title": "Photosynthesis Equation",
        "steps": [
            ("Balanced equation for photosynthesis.",
             "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. The reverse of cellular respiration."),
            ("Photosynthesis occurs in the chloroplasts. The light reactions capture energy, the Calvin cycle fixes carbon.",
             "It is the primary source of both organic matter and atmospheric oxygen on Earth."),
        ],
    },
    {
        "title": "Protein Synthesis",
        "steps": [
            ("Briefly describe the central dogma of molecular biology.",
             "DNA is transcribed into mRNA in the nucleus. mRNA carries the code to ribosomes. tRNA brings amino acids matching mRNA codons."),
            ("The ribosome links amino acids into a polypeptide chain. This is translation. Flow: DNA -> RNA -> protein.",
             "The central dogma is one-directional for most cellular processes."),
        ],
    },
    {
        "title": "Natural Selection",
        "steps": [
            ("Three necessary conditions for natural selection.",
             "Variation: individuals differ in traits. Heritability: differences are inherited. Differential reproduction: some variants survive and reproduce more."),
            ("When these conditions hold, advantageous traits become more common over generations. This drives adaptation.",
             "Evolution by natural selection was Darwin's great insight."),
        ],
    },
    {
        "title": "Blood Types",
        "steps": [
            ("What are the four blood types and their compatibility?",
             "Type A: A antigens, anti-B antibodies. Type B: B antigens, anti-A antibodies. Type AB: both antigens, no antibodies (universal recipient). Type O: no antigens, both antibodies (universal donor)."),
            ("O can donate to all. AB can receive from all. A receives from A and O. B receives from B and O.",
             "Mismatched transfusions cause clumping and can be fatal."),
        ],
    },
    {
        "title": "Punnett Square",
        "steps": [
            ("Two heterozygous pea plants (Tt) crossed. T = tall, t = short. Genotype and phenotype ratios?",
             "Punnett: Tt x Tt. Gametes: T and t. Offspring: TT, Tt, tT, tt."),
            ("Genotype: 1 TT : 2 Tt : 1 tt. Phenotype: 3 tall : 1 short.",
             "Mendel's classic monohybrid cross. The 3:1 ratio was key to discovering dominance."),
        ],
    },
    {
        "title": "Enzyme Function",
        "steps": [
            ("How do enzymes speed up reactions?",
             "Enzymes are biological catalysts. They lower the activation energy of reactions by binding substrates at their active site."),
            ("The induced fit model: the enzyme changes shape on binding, stressing bonds and facilitating the reaction. Enzymes are not consumed.",
             "Without enzymes, most biochemical reactions would be impractically slow."),
        ],
    },
    {
        "title": "Food Chain Pyramid",
        "steps": [
            ("Why is energy transfer between trophic levels only about 10% efficient?",
             "Energy is lost as heat through respiration, used for metabolism, or egested as waste. Only energy incorporated into biomass passes to the next level."),
            ("This explains why food chains rarely exceed 4-5 levels and why top predators are rare. About 90% of energy is lost at each step.",
             "It also explains why there is more biomass at lower trophic levels."),
        ],
    },
    # ---- ECONOMICS ----
    {
        "title": "Supply and Demand",
        "steps": [
            ("If coffee prices rise, what happens to tea demand?",
             "Coffee and tea are substitutes. When coffee gets more expensive, consumers switch to the relatively cheaper alternative."),
            ("Demand for tea increases, shifting its demand curve right, raising equilibrium price and quantity for tea.",
             "Cross-price elasticity measures this relationship between substitute goods."),
        ],
    },
    {
        "title": "Price Elasticity of Demand",
        "steps": [
            ("10% price increase causes 20% drop in quantity. PED?",
             "PED = % change in quantity / % change in price = -20% / +10% = -2. Absolute value is 2."),
            ("|PED| > 1, so demand is elastic. Price increase leads to lower total revenue because the quantity drop is proportionally larger.",
             "Elastic goods have many substitutes or are luxury items."),
        ],
    },
    {
        "title": "Opportunity Cost",
        "steps": [
            ("$50 concert ticket, could have worked 4 hours at $15/hour. Opportunity cost?",
             "Explicit: $50 ticket. Implicit: 4 * $15 = $60 in foregone wages."),
            ("Total opportunity cost = $50 + $60 = $110. This is the true cost of attending the concert.",
             "Opportunity cost includes both explicit outlays and forgone alternatives."),
        ],
    },
    {
        "title": "Comparative Advantage",
        "steps": [
            ("Country A: 10 cars or 5 computers per worker. Country B: 8 cars or 4 computers per worker. Should they trade?",
             "A's opportunity cost of 1 car = 0.5 computers. B's opportunity cost of 1 car = 0.5 computers. Both have the same ratio."),
            ("No comparative advantage exists. Trade does not provide gains since opportunity cost ratios are identical.",
             "Absolute advantage is different from comparative advantage. Trade gains come from differing ratios."),
        ],
    },
    {
        "title": "Inflation Calculation",
        "steps": [
            ("Basket costs $100 in 2020, $110 in 2021. Inflation rate?",
             "Inflation = ((CPI_new - CPI_old) / CPI_old) * 100% = ((110 - 100) / 100) * 100%."),
            ("Inflation rate = 10%. That is quite high by modern standards.",
             "Central banks typically target 2-3% annual inflation."),
        ],
    },
    {
        "title": "Compound Interest",
        "steps": [
            ("$5000 at 6% compounded monthly for 2 years. Future value?",
             "A = P(1+r/n)^(nt) = 5000 * (1 + 0.06/12)^(24) = 5000 * (1.005)^24."),
            ("(1.005)^24 = 1.12716. A = 5000 * 1.12716 = $5635.80.",
             "Compound interest grows faster than simple because interest earns interest."),
        ],
    },
    {
        "title": "GDP Calculation",
        "steps": [
            ("What are the four components of GDP?",
             "GDP = C + I + G + NX. Consumption (household spending), Investment (business capital), Government spending, Net Exports (exports - imports)."),
            ("Consumption is typically the largest component in developed economies, often around 60-70% of GDP.",
             "GDP measures total output within a country's borders, regardless of who owns the factors of production."),
        ],
    },
    {
        "title": "Law of Diminishing Marginal Utility",
        "steps": [
            ("Explain why the first slice of pizza is more satisfying than the fifth.",
             "Marginal utility decreases with each additional unit consumed. The first slice satisfies hunger; later slices provide less additional satisfaction."),
            ("This explains the downward-sloping demand curve: consumers buy more only at lower prices because marginal utility diminishes.",
             "Total utility still increases as long as marginal utility is positive."),
        ],
    },
    {
        "title": "Market Equilibrium",
        "steps": [
            ("Qd = 100 - 2P, Qs = 3P. Find equilibrium.",
             "At equilibrium Qd = Qs: 100 - 2P = 3P -> 100 = 5P -> P = 20."),
            ("Q = 3 * 20 = 60. Equilibrium price is $20, quantity is 60 units.",
             "At any other price, there would be a surplus or shortage, creating pressure to return to equilibrium."),
        ],
    },
    {
        "title": "Consumer Surplus",
        "steps": [
            ("Demand: P = 50 - Q, price = $30. Consumer surplus?",
             "At P = 30, Q = 50 - 30 = 20. Consumer surplus is the triangle below demand and above price."),
            ("CS = 1/2 * 20 * (50-30) = 1/2 * 20 * 20 = 200.",
             "Consumer surplus measures the benefit consumers receive above what they pay."),
        ],
    },
    # ---- LANGUAGE / LINGUISTICS ----
    {
        "title": "Parts of Speech",
        "steps": [
            ("Identify parts of speech in: 'The quick brown fox jumps over the lazy dog.'",
             "The (article), quick (adjective), brown (adjective), fox (noun), jumps (verb), over (preposition), the (article), lazy (adjective), dog (noun)."),
            ("The sentence has 1 verb (jumps), 2 nouns (fox, dog), 3 adjectives (quick, brown, lazy), 1 preposition (over), and 2 articles (the, the).",
             "This sentence contains every letter of the alphabet — it is a pangram."),
        ],
    },
    {
        "title": "Sentence Structure",
        "steps": [
            ("Identify subject, verb, object in: 'The cat chased the mouse.'",
             "Subject: The cat (performer). Verb: chased (action). Object: the mouse (receiver)."),
            ("This is an SVO (Subject-Verb-Object) sentence. English typically follows this order.",
             "Other languages use SOV (Japanese) or VSO (Irish) patterns."),
        ],
    },
    {
        "title": "Prefix and Suffix",
        "steps": [
            ("Break down 'unbelievable' into morphemes.",
             "un- (prefix: 'not') + believe (root) + -able (suffix: 'capable of')."),
            ("'Unbelievable' means 'not capable of being believed.' Morphemes are the smallest meaning-bearing units.",
             "English morphology is both derivational (creating new words) and inflectional (modifying grammar)."),
        ],
    },
    {
        "title": "Active vs Passive Voice",
        "steps": [
            ("Convert 'The ball was thrown by the boy' from passive to active.",
             "In passive, the subject receives the action. Make the agent 'the boy' the subject and the receiver 'the ball' the object."),
            ("Active: 'The boy threw the ball.' The active voice is typically more direct and concise.",
             "Passive is useful when the agent is unknown or when we want to emphasize the recipient."),
        ],
    },
    {
        "title": "Synonyms and Antonyms",
        "steps": [
            ("Find synonyms and antonyms for 'benevolent'.",
             "Synonyms: kind, generous, charitable. Antonyms: malevolent, cruel, unkind."),
            ("Benevolent comes from Latin bene (well) + volent (wishing) — literally 'well-wishing.' Understanding etymology helps remember meanings.",
             "Synonyms share meaning but are rarely perfectly interchangeable due to connotation differences."),
        ],
    },
    {
        "title": "Common Etymologies",
        "steps": [
            ("Origin of the word 'etymology'?",
             "From Greek etymon (true sense) + logos (word/study). It entered English through Old French etimologie in the 14th century."),
            ("Etymology is the study of the true origin and historical development of words.",
             "English borrows heavily from Latin, French, and Germanic sources, giving it a rich vocabulary."),
        ],
    },
    {
        "title": "Figurative Language",
        "steps": [
            ("Identify the figure of speech: 'The world is a stage.'",
             "This is a metaphor — a direct comparison without 'like' or 'as.' The world is not literally a stage but is being compared to one for dramatic effect."),
            ("If it said 'like a stage,' it would be a simile. This is from Shakespeare's As You Like It.",
             "Metaphors are pervasive in everyday language, not just poetry."),
        ],
    },
    {
        "title": "Grammar: Subject-Verb Agreement",
        "steps": [
            ("Correct: 'The group of students is/are meeting'?",
             "The subject is 'group' (singular), not 'students' (object of the preposition). Prepositional phrases do not change the number of the subject."),
            ("Correct: 'The group of students is meeting.' The singular verb agrees with the singular subject.",
             "Subject-verb agreement errors often occur when a plural noun intervenes between subject and verb."),
        ],
    },
    {
        "title": "Phonetics: Voiced vs Voiceless",
        "steps": [
            ("Are /b/ and /p/ voiced or voiceless?",
             "/b/ is voiced (vocal cords vibrate). /p/ is voiceless (no vibration). Both are bilabial plosives, differing only in voicing."),
            ("This is a minimal pair: 'bat' vs 'pat' differ in the initial phoneme's voicing. Voicing is a distinctive feature in English.",
             "Other languages, like Thai, distinguish more than two degrees of aspiration."),
        ],
    },
    {
        "title": "Semantic Field",
        "steps": [
            ("Identify the semantic field of: surgeon, scalpel, stethoscope, ward, diagnosis.",
             "All relate to medicine. Surgeon (professional), scalpel/stethoscope (instruments), ward (location), diagnosis (process)."),
            ("These words share a semantic field, allowing them to co-occur in medical discourse. Semantic fields reveal conceptual organization.",
             "Words in the same semantic field often share collocations and grammatical patterns."),
        ],
    },
    # ---- PHILOSOPHY / LOGIC ----
    {
        "title": "Deductive vs Inductive Reasoning",
        "steps": [
            ("Identify reasoning type: 'All humans are mortal. Socrates is human. Therefore, Socrates is mortal.'",
             "This is deduction: the conclusion necessarily follows from the premises. If premises are true, conclusion must be true."),
            ("Deduction guarantees the conclusion. Induction (e.g., 'all observed swans are white, so all swans are white') is probabilistic, not guaranteed.",
             "Valid deduction preserves truth from premises to conclusion."),
        ],
    },
    {
        "title": "Logical Fallacy: Straw Man",
        "steps": [
            ("Identify the fallacy: 'A: We should improve schools. B: So you want to raise taxes and bankrupt everyone?'",
             "This is a straw man. B misrepresents A's moderate position as extreme, then attacks the distorted version."),
            ("The straw man is a common rhetorical tactic. The proper response is to point out the misrepresentation and restate the original position.",
             "Other common fallacies include ad hominem, false dilemma, and slippery slope."),
        ],
    },
    {
        "title": "The Trolley Problem",
        "steps": [
            ("Trolley heading toward five people. Pull lever to divert to side track with one person. Do you pull?",
             "Utilitarian: maximize well-being, pull the lever (save five at cost of one). Deontological: actively causing death is wrong, do not intervene."),
            ("The trolley problem reveals the tension between consequentialist and deontological ethics. There is no universally agreed answer.",
             "It is a thought experiment, not a practical dilemma. It probes moral intuitions."),
        ],
    },
    {
        "title": "Cogito Ergo Sum",
        "steps": [
            ("What is Descartes' 'Cogito, ergo sum' and what does it mean?",
             "'I think, therefore I am.' Descartes used this as the indubitable foundation after methodological doubt, doubting everything that could possibly be doubted."),
            ("Even if an evil demon deceives me about everything, the act of being deceived requires a thinker. The thinking self's existence is certain.",
             "This established the subject as the starting point for modern Western philosophy."),
        ],
    },
    {
        "title": "Plato's Allegory of the Cave",
        "steps": [
            ("Summarize Plato's Allegory of the Cave.",
             "Prisoners in a cave see only shadows on the wall, believing these are reality. One escapes, sees the real world, and realizes the shadows were mere projections."),
            ("The allegory represents the philosopher's journey from ignorance to knowledge of the Forms. It illustrates both enlightenment and the resistance of those still in the cave.",
             "Plato's epistemology is deeply shaped by this distinction between appearance and reality."),
        ],
    },
    {
        "title": "The Problem of Evil",
        "steps": [
            ("What is the logical problem of evil?",
             "If God is omnipotent, omniscient, and omnibenevolent, then evil should not exist. But evil does exist. Therefore, such a God cannot exist."),
            ("Responses: the free will defense (evil results from free choice), soul-making theodicy (evil builds character), and the epistemic distance argument.",
             "This is one of the most persistent challenges to classical theism."),
        ],
    },
    {
        "title": "Occam's Razor",
        "steps": [
            ("Explain Occam's Razor and its use.",
             "Occam's Razor: among competing hypotheses, the one with the fewest assumptions should be preferred. 'Entities should not be multiplied without necessity.'"),
            ("It is a heuristic, not a proof. Simpler theories are easier to test and falsify. It guides scientific theory choice toward parsimony.",
             "Occam's Razor does not mean the simplest theory is always true, just that it is the best starting point."),
        ],
    },
    {
        "title": "The Ship of Theseus",
        "steps": [
            ("If all parts of a ship are replaced over time, is it still the same ship?",
             "This is a paradox about identity over time. If the original planks are reassembled, which vessel is the original — the continuously replaced one or the reassembled one?"),
            ("The paradox explores whether identity is tied to material composition, spatiotemporal continuity, or form. It applies to personal identity as well.",
             "Every cell in your body is replaced over years. Are you still the same person?"),
        ],
    },
    {
        "title": "A Priori vs A Posteriori",
        "steps": [
            ("Classify: 'All bachelors are unmarried' as a priori or a posteriori.",
             "A priori knowledge is independent of experience. This statement is true by definition — 'bachelor' includes 'unmarried.' It requires no empirical investigation."),
            ("It is a priori, analytic knowledge: true solely by virtue of meaning. A posteriori knowledge (e.g., 'it is raining') requires empirical observation.",
             "Kant famously asked whether synthetic a priori knowledge is possible."),
        ],
    },
    {
        "title": "The Veil of Ignorance",
        "steps": [
            ("Explain Rawls' Veil of Ignorance.",
             "Rawls asks us to design a just society from behind a veil where we do not know our own position (wealth, talents, gender, etc.) in that society."),
            ("From this impartial perspective, rational people would choose principles that protect the worst-off — the difference principle. This yields justice as fairness.",
             "The veil of ignorance operationalizes impartiality in political philosophy."),
        ],
    },
    # ---- GEOGRAPHY / GENERAL ----
    {
        "title": "Latitude and Longitude",
        "steps": [
            ("What is the significance of the Prime Meridian?",
             "The Prime Meridian (0 degrees longitude) runs through Greenwich, England. It divides the Earth into Eastern and Western hemispheres."),
            ("Established as the international standard in 1884. Longitude is measured east or west up to 180 degrees.",
             "Latitude measures north-south position relative to the equator."),
        ],
    },
    {
        "title": "Population Density",
        "steps": [
            ("City with 500,000 people and 200 square km. Population density?",
             "Density = population / area = 500,000 / 200."),
            ("Density = 2,500 people per square kilometer.",
             "That is moderately dense. Compare to Mumbai at about 30,000 per km^2."),
        ],
    },
    {
        "title": "Scale Factor",
        "steps": [
            ("Map scale 1:100,000. Two cities 5 cm apart. Actual distance?",
             "1 cm = 100,000 cm = 1 km. 5 cm = 500,000 cm."),
            ("500,000 cm = 5,000 m = 5 km.",
             "Map scales are ratios. 1:100,000 means one unit on the map = 100,000 units on the ground."),
        ],
    },
    {
        "title": "Temperature Conversion",
        "steps": [
            ("Convert 25 C to Fahrenheit.",
             "F = C * 9/5 + 32 = 25 * 9/5 + 32 = 45 + 32."),
            ("25 C = 77 F. This is a warm room temperature.",
             "The conversion formulas are linear relationships between the two scales."),
        ],
    },
    {
        "title": "Significant Figures",
        "steps": [
            ("How many significant figures in 0.00450?",
             "Leading zeros are not significant. Trailing zeros after the decimal are significant. Digits: 4, 5, and the trailing 0."),
            ("0.00450 has 3 significant figures. The leading zeros only indicate place value.",
             "Significant figures communicate the precision of a measurement."),
        ],
    },
    {
        "title": "Scientific Notation",
        "steps": [
            ("Write 0.000000123 in scientific notation.",
             "Move the decimal 7 places to the right to get 1.23. The exponent is -7 (right shift = negative)."),
            ("0.000000123 = 1.23 * 10^(-7). Scientific notation makes very small and very large numbers manageable.",
             "The exponent tells you how many places to shift the decimal."),
        ],
    },
    {
        "title": "Percentage Change",
        "steps": [
            ("Stock drops from $80 to $60. Percentage decrease?",
             "Percentage change = (new - old) / old * 100% = (60 - 80) / 80 * 100% = -20/80 * 100%."),
            ("-25%, or a 25% decrease. A $20 drop from an $80 base is one quarter.",
             "Percentage change is relative to the original value, not the new value."),
        ],
    },
    {
        "title": "Simple Interest",
        "steps": [
            ("$2000 at 5% simple annual interest for 3 years. Total interest?",
             "I = P * r * t = 2000 * 0.05 * 3."),
            ("Interest = $300. Total repayment = $2,300. Simple interest does not compound — it is linear.",
             "Most loans use compound interest, which grows faster."),
        ],
    },
    {
        "title": "Rate of Change",
        "steps": [
            ("Plant grows from 10 cm to 25 cm over 5 weeks. Average growth rate?",
             "Average rate = (change in height) / (change in time) = (25 - 10) / 5."),
            ("15/5 = 3 cm per week. This is the slope of the line connecting the start and end points.",
             "Rate of change measures how one quantity changes relative to another."),
        ],
    },
    {
        "title": "Ratio and Proportion",
        "steps": [
            ("3 apples cost $1.50. Cost of 7 apples?",
             "Unit price = 1.50 / 3 = $0.50 per apple. 7 apples = 7 * 0.50."),
            ("7 apples cost $3.50. Proportions assume a constant unit rate.",
             "Cross-multiplication also works: 3/1.50 = 7/x -> 3x = 10.50 -> x = 3.50."),
        ],
    },
    # ---- MORE MATH ----
    {
        "title": "Derivative of a Polynomial",
        "steps": [
            ("Find derivative of f(x) = 3x^2 + 2x - 5.",
             "Power rule: d/dx(x^n) = nx^(n-1). 3x^2 -> 6x. 2x -> 2. -5 (constant) -> 0."),
            ("f'(x) = 6x + 2. The derivative gives the slope of the tangent line at any point.",
             "Differentiation is a linear operator: the derivative of a sum is the sum of derivatives."),
        ],
    },
    {
        "title": "Integration",
        "steps": [
            ("Indefinite integral of 2x dx.",
             "Power rule in reverse: integral of x^n dx = x^(n+1)/(n+1) + C. Integral of 2x = x^2 + C."),
            ("Check: d/dx(x^2 + C) = 2x. The +C accounts for the constant of integration.",
             "Indefinite integrals represent families of functions differing by a constant."),
        ],
    },
    {
        "title": "Definite Integral",
        "steps": [
            ("Evaluate integral of x from 0 to 4.",
             "Antiderivative of x is x^2/2. Evaluate: (4^2/2) - (0^2/2) = 8 - 0."),
            ("The area under y = x from 0 to 4 is 8. This is a right triangle with base 4 and height 4, area = 1/2*4*4 = 8.",
             "The Fundamental Theorem of Calculus links differentiation and integration."),
        ],
    },
    {
        "title": "Chain Rule",
        "steps": [
            ("Derivative of h(x) = (3x^2 + 1)^5.",
             "Let u = 3x^2 + 1. Then h = u^5. Chain rule: dh/dx = dh/du * du/dx = 5u^4 * 6x."),
            ("h'(x) = 5(3x^2+1)^4 * 6x = 30x(3x^2+1)^4. The chain rule composes derivatives.",
             "The chain rule is essential for differentiating composite functions."),
        ],
    },
    {
        "title": "Limits",
        "steps": [
            ("Limit as x->2 of (x^2-4)/(x-2).",
             "Direct substitution gives 0/0, indeterminate. Factor: (x-2)(x+2)/(x-2) = x+2, for x ≠ 2."),
            ("Limit = 2+2 = 4. The function has a removable discontinuity at x = 2.",
             "Factoring, rationalizing, or L'Hospital's rule resolve indeterminate forms."),
        ],
    },
    {
        "title": "Absolute Value Equation",
        "steps": [
            ("Solve |x - 3| = 5.",
             "Absolute value means distance from zero. x-3 = 5 or x-3 = -5."),
            ("x = 8 or x = -2. Both satisfy |x-3| = 5.",
             "Absolute value equations typically have two solutions."),
        ],
    },
    {
        "title": "Inequality Solving",
        "steps": [
            ("Solve 2x - 3 < 7.",
             "Add 3: 2x < 10. Divide by 2: x < 5."),
            ("x < 5 or (-infinity, 5) in interval notation. The inequality sign stays the same when multiplying/dividing by a positive number.",
             "Multiplying by a negative flips the inequality sign."),
        ],
    },
    {
        "title": "Midpoint Formula",
        "steps": [
            ("Midpoint between (2, 3) and (8, 7).",
             "Midpoint = ((x1+x2)/2, (y1+y2)/2) = ((2+8)/2, (3+7)/2) = (10/2, 10/2)."),
            ("Midpoint = (5, 5). It is the point exactly halfway between the two given points.",
             "The midpoint formula averages the coordinates."),
        ],
    },
    {
        "title": "Distance Formula",
        "steps": [
            ("Distance between (1, 2) and (4, 6).",
             "Distance = sqrt((x2-x1)^2 + (y2-y1)^2) = sqrt(3^2 + 4^2) = sqrt(9+16)."),
            ("Distance = sqrt(25) = 5. This is the Pythagorean theorem applied to coordinate differences.",
             "The 3-4-5 right triangle is the simplest Pythagorean triple."),
        ],
    },
    {
        "title": "Equation of a Circle",
        "steps": [
            ("Circle with center (3, -2) and radius 4. Write equation.",
             "Standard form: (x - h)^2 + (y - k)^2 = r^2 with center (h,k)."),
            ("(x - 3)^2 + (y + 2)^2 = 16. The radius squared goes on the right side.",
             "The equation describes all points at distance r from the center."),
        ],
    },
    {
        "title": "Direct Variation",
        "steps": [
            ("y varies directly with x. y = 12 when x = 4. Find y when x = 7.",
             "y = kx. 12 = k*4 -> k = 3. So y = 3x."),
            ("When x = 7: y = 3*7 = 21. Direct variation means y/x is constant.",
             "The graph of a direct variation is a straight line through the origin."),
        ],
    },
    {
        "title": "Inverse Variation",
        "steps": [
            ("y varies inversely with x. y = 6 when x = 8. Find y when x = 12.",
             "y = k/x. 6 = k/8 -> k = 48. So y = 48/x."),
            ("y = 48/12 = 4. Inverse variation means x*y is constant.",
             "The graph of an inverse variation is a hyperbola."),
        ],
    },
    {
        "title": "Sequence: Next Term",
        "steps": [
            ("Next term in 1, 1, 2, 3, 5, 8, 13, ...?",
             "This is the Fibonacci sequence: each term is the sum of the two preceding terms. 8+5=13, 13+8=21."),
            ("The next term is 21. The Fibonacci sequence appears throughout nature — in pinecones, sunflowers, and nautilus shells.",
             "It was introduced to Western mathematics by Leonardo of Pisa (Fibonacci) in 1202."),
        ],
    },
    {
        "title": "Unit Conversion: Metric",
        "steps": [
            ("Convert 2500 meters to kilometers.",
             "1 km = 1000 m. 2500 / 1000 = 2.5."),
            ("2500 m = 2.5 km. The metric system is based on powers of ten, making conversions straightforward.",
             "Kilo- means 1000, so converting meters to kilometers just requires dividing by 1000."),
        ],
    },
    {
        "title": "Unit Conversion: Imperial",
        "steps": [
            ("Convert 3 miles to feet.",
             "1 mile = 5280 feet. 3 * 5280 = 15,840."),
            ("3 miles = 15,840 feet. Imperial conversions use less intuitive factors than metric.",
             "The US customary system uses 5280 feet per mile, which comes from Roman and English surveying tradition."),
        ],
    },
    {
        "title": "Perimeter of a Rectangle",
        "steps": [
            ("Rectangle with length 12 cm and width 8 cm. Perimeter?",
             "P = 2*(l+w) = 2*(12+8) = 2*20."),
            ("P = 40 cm. Perimeter is the total distance around the shape.",
             "Area would be l*w = 96 cm^2 — different from perimeter."),
        ],
    },
    {
        "title": "Area of a Triangle",
        "steps": [
            ("Triangle with base 10 cm and height 6 cm. Area?",
             "A = 1/2 * b * h = 1/2 * 10 * 6."),
            ("A = 30 square cm. The formula works for any triangle — right, acute, or obtuse.",
             "The height must be perpendicular to the base for this formula."),
        ],
    },
    {
        "title": "Surface Area of a Cube",
        "steps": [
            ("Cube with side length 5 cm. Surface area?",
             "Each face has area 25 cm^2. A cube has 6 faces."),
            ("SA = 6 * 25 = 150 cm^2. Volume would be 5^3 = 125 cm^3.",
             "Surface area grows with the square of side length, volume with the cube."),
        ],
    },
    {
        "title": "Volume of a Rectangular Prism",
        "steps": [
            ("Box measuring 3 m by 4 m by 5 m. Volume?",
             "V = l * w * h = 3 * 4 * 5 = 60."),
            ("V = 60 cubic meters. Volume measures three-dimensional space occupancy.",
             "This is 60,000 liters of capacity."),
        ],
    },
    {
        "title": "Circumference of a Circle",
        "steps": [
            ("Circle with diameter 14 cm. Circumference? (pi = 22/7)",
             "C = pi * d = (22/7) * 14 = 22 * 2."),
            ("C = 44 cm. The ratio of circumference to diameter is always pi.",
             "Pi is an irrational number approximately equal to 3.14159..."),
        ],
    },
    # ---- MORE PHYSICS ----
    {
        "title": "Pressure Calculation",
        "steps": [
            ("100 N force over 0.5 m^2. Pressure?",
             "P = F/A = 100 / 0.5."),
            ("P = 200 Pa. If the same force were applied over a smaller area, pressure would increase.",
             "This is why sharp objects cut more easily — the same force concentrated on a smaller area."),
        ],
    },
    {
        "title": "Density",
        "steps": [
            ("Object mass 300 g, volume 150 cm^3. Density?",
             "Density = mass / volume = 300 / 150."),
            ("Density = 2 g/cm^3. About twice the density of water. It would sink.",
             "Density is an intensive property — it does not depend on the amount of material."),
        ],
    },
    {
        "title": "Speed of Sound",
        "steps": [
            ("Thunder heard 3 seconds after lightning. Distance? (v = 343 m/s)",
             "d = v * t = 343 * 3 = 1029 m."),
            ("About 1 km. Counting seconds between flash and thunder gives distance in kilometers: 3 seconds ≈ 1 km.",
             "Light travels so fast (3*10^8 m/s) that its travel time is negligible here."),
        ],
    },
    {
        "title": "Refraction Index",
        "steps": [
            ("Light travels at 2.25*10^8 m/s in a material. Refractive index? (c = 3*10^8 m/s)",
             "n = c/v = 3*10^8 / 2.25*10^8."),
            ("n = 1.33, matching water. Higher refractive index means light slows down more.",
             "This is why a straw appears bent in a glass of water."),
        ],
    },
    {
        "title": "Specific Heat Capacity",
        "steps": [
            ("500 J raises 100 g of substance by 10 C. Specific heat?",
             "c = Q/(m*delta_T) = 500/(0.1*10) = 500 J/(kg*K)."),
            ("c = 500 J/(kg*K). This is lower than water's 4180, so it heats up more easily.",
             "Substances with low specific heat change temperature quickly."),
        ],
    },
    {
        "title": "Coulomb's Law",
        "steps": [
            ("Two 2*10^(-6) C charges 0.3 m apart. Force? (k = 9*10^9 N*m^2/C^2)",
             "F = k*q1*q2/r^2 = 9*10^9 * (2*10^(-6))^2 / 0.3^2."),
            ("F = 9*10^9 * 4*10^(-12) / 0.09 = 36*10^(-3)/0.09 = 0.4 N.",
             "Coulomb's law is structurally identical to Newton's law of gravitation."),
        ],
    },
    {
        "title": "Doppler Effect",
        "steps": [
            ("500 Hz source approaches observer at 30 m/s. Observed frequency? (v = 340 m/s)",
             "f' = f * v/(v - v_source) = 500 * 340 / (340 - 30) = 500 * 340/310."),
            ("f' = 500 * 1.097 = 548.4 Hz. The observer hears a higher pitch as the source approaches.",
             "The formula changes depending on whether the source or observer is moving."),
        ],
    },
    {
        "title": "Momentum Impulse",
        "steps": [
            ("0.5 kg ball at 10 m/s stopped in 0.1 s. Average force?",
             "Impulse = change in momentum = 0.5*10 = 5 N*s. F = impulse/t = 5/0.1."),
            ("F = 50 N. Stopping the same ball in less time would require more force.",
             "This is why airbags increase collision time to reduce force."),
        ],
    },
    {
        "title": "Rotational Kinetic Energy",
        "steps": [
            ("Disk with I = 0.02 kg*m^2 rotates at 10 rad/s. Rotational KE?",
             "RKE = 1/2 * I * omega^2 = 0.5 * 0.02 * 100."),
            ("RKE = 1 J. Rotational kinetic energy is analogous to linear kinetic energy with I replacing m and omega replacing v.",
             "Moment of inertia depends on how mass is distributed relative to the axis."),
        ],
    },
    {
        "title": "Thermal Expansion",
        "steps": [
            ("Steel rail 20 m at 10 C. Expansion at 40 C? (alpha = 1.2*10^(-5)/C)",
             "delta_L = L0*alpha*delta_T = 20 * 1.2*10^(-5) * 30 = 20 * 3.6*10^(-4)."),
            ("delta_L = 0.0072 m = 7.2 mm. Expansion gaps in railroad tracks must accommodate this change.",
             "Different materials expand at different rates, which is why bimetallic strips bend when heated."),
        ],
    },
    # ---- MORE PROGRAMMING ----
    {
        "title": "Stack Operations",
        "steps": [
            ("Evaluate postfix: 5 3 + 8 *.",
             "Push 5. Push 3. Pop 3,5 -> 5+3=8, push 8. Push 8. Pop 8,8 -> 8*8=64, push 64."),
            ("Result: 64. Postfix notation eliminates parentheses because operator order is unambiguous.",
             "It is also called Reverse Polish Notation (RPN) and is evaluation-efficient with a stack."),
        ],
    },
    {
        "title": "Queue vs Stack",
        "steps": [
            ("What is the difference between a stack and a queue?",
             "Stack: LIFO (Last In First Out) — push/pop at one end, like a pile of plates. Queue: FIFO (First In First Out) — enqueue at rear, dequeue at front, like a line of people."),
            ("Stacks are used for function call management, undo operations. Queues are used for breadth-first search, print spooling.",
             "Both are fundamental abstract data types in computer science."),
        ],
    },
    {
        "title": "Recursive vs Iterative",
        "steps": [
            ("Compare recursion and iteration for factorial.",
             "Recursive: f(n) = n*f(n-1), base f(1)=1. Iterative: loop 1..n accumulating product. Both compute the same result."),
            ("Recursion is elegant but may cause stack overflow for large n. Iteration uses constant stack space. In practice, iteration is preferred for deep computations.",
             "Tail recursion can be optimized to iteration by the compiler."),
        ],
    },
    {
        "title": "Hash Table Collision",
        "steps": [
            ("Explain chaining for hash table collisions.",
             "Chaining: each bucket stores a linked list of key-value pairs. On collision, append to the list. On lookup, traverse the list to find the key."),
            ("Open addressing: on collision, probe the next available slot (linear, quadratic, or double hashing). Chaining is simpler but uses more memory per slot.",
             "Good hash functions minimize collisions, making both approaches efficient."),
        ],
    },
    {
        "title": "Big-O for Common Operations",
        "steps": [
            ("Insert at beginning of array vs linked list complexity?",
             "Array: O(n) because all elements must shift right. Linked list: O(1) because only the head pointer is updated."),
            ("Linked lists excel at head insertions/deletions but have O(n) random access. Arrays offer O(1) random access.",
             "Each structure optimizes for different access patterns."),
        ],
    },
    {
        "title": "Selection Sort",
        "steps": [
            ("Trace selection sort on [29, 10, 14, 37, 13].",
             "Find min (10) at 1, swap with index 0: [10, 29, 14, 37, 13]. Find min in [1..4] (13 at 4), swap with 1: [10, 13, 14, 37, 29]."),
            ("Find min in [2..4] (14 at 2), already in place. Find min in [3..4] (29 at 4), swap: [10, 13, 14, 29, 37]. Sorted in O(n^2).",
             "Selection sort always does O(n^2) comparisons regardless of input order."),
        ],
    },
    {
        "title": "Insertion Sort",
        "steps": [
            ("Trace insertion sort on [8, 3, 5, 1].",
             "[8]. Insert 3: compare with 8, shift right, insert: [3,8,5,1]. Insert 5: compare with 8, shift, compare with 3, insert: [3,5,8,1]."),
            ("Insert 1: compare with 8,5,3, shift all, insert: [1,3,5,8]. Insertion sort is O(n^2) worst case but O(n) best case on nearly sorted data.",
             "Insertion sort is often used for small arrays in hybrid sorting algorithms."),
        ],
    },
    {
        "title": "Merge Sort Concept",
        "steps": [
            ("Explain the divide step of merge sort on [38, 27, 43, 3, 9, 82, 10].",
             "Split into [38,27,43,3] and [9,82,10]. Split: [38,27], [43,3], [9,82], [10]. Split: [38],[27],[43],[3],[9],[82],[10]."),
            ("Then merge back in sorted order. Merge sort is O(n log n) in all cases, making it more predictable than quicksort.",
             "It requires O(n) auxiliary space for the merge step."),
        ],
    },
    {
        "title": "Tree Traversal: Inorder",
        "steps": [
            ("Inorder traversal of tree with root 4, left [2 with leaves 1,3], right [5 with right 6].",
             "Inorder: visit left subtree (1,2,3), then root (4), then right subtree (5,6)."),
            ("Inorder: 1, 2, 3, 4, 5, 6. Inorder traversal of a BST yields elements in sorted order.",
             "Preorder visits root first, postorder visits root last."),
        ],
    },
    {
        "title": "Bubble Sort Optimization",
        "steps": [
            ("How can bubble sort be optimized?",
             "Track whether any swaps occurred in a pass. If no swaps, the array is sorted and we can terminate early. This makes best-case O(n). Also, after each pass the largest remaining element is at its final position, so the inner loop can shrink by one each iteration."),
        ],
    },
    # ---- MORE LOGIC PUZZLES ----
    {
        "title": "Syllogism Validity",
        "steps": [
            ("Is this syllogism valid? All mammals have fur. All dogs are mammals. Therefore, all dogs have fur.",
             "Premise 1: All M are F. Premise 2: All D are M. Conclusion: All D are F. This follows the Barbara syllogism form."),
            ("It is valid: if the premises are true, the conclusion must be true. Validity is about form, not factual truth.",
             "A syllogism can be valid but unsound if a premise is false."),
        ],
    },
    {
        "title": "Confirmation Bias",
        "steps": [
            ("Define confirmation bias.",
             "Confirmation bias is the tendency to search for, interpret, favor, and recall information that confirms one's preexisting beliefs while giving disproportionately less weight to contrary evidence."),
            ("It is a common cognitive bias that can lead to poor decisions. Mitigation includes actively seeking disconfirming evidence and considering alternative hypotheses.",
             "It was first experimentally demonstrated by Peter Wason in the 1960s."),
        ],
    },
    {
        "title": "Gambler's Fallacy",
        "steps": [
            ("What is the gambler's fallacy?",
             "The gambler's fallacy is the mistaken belief that if an event has occurred more frequently than usual, it is less likely to occur in the future (or vice versa), when the events are independent."),
            ("Example: after 5 coin flips landing heads, believing tails is 'due.' Each flip has a 50% chance regardless of past outcomes. The coin has no memory.",
             "This fallacy arises from misunderstanding the law of large numbers."),
        ],
    },
    {
        "title": "The Halo Effect",
        "steps": [
            ("Explain the halo effect with an example.",
             "The halo effect is a cognitive bias where a positive impression in one area (e.g., physical attractiveness) influences opinion in unrelated areas (e.g., intelligence, kindness)."),
            ("This bias affects hiring, grading, and marketing. Blind auditions in orchestras were introduced specifically to counter it.",
             "The term was first coined by psychologist Edward Thorndike in 1920."),
        ],
    },
    {
        "title": "Base Rate Fallacy",
        "steps": [
            ("99% accurate test for a disease with 1% prevalence. You test positive. Probability you actually have it?",
             "In 10,000 people: 100 have the disease, 99 test positive. 9,900 do not, 99 test positive (false positives). Total positives: 198."),
            ("P(disease|positive) = 99/198 = 50%. Ignoring the base rate would lead to overestimating this probability dramatically.",
             "Bayes' theorem formalizes how prior probabilities should be updated with new evidence."),
        ],
    },
    {
        "title": "Correlation vs Causation",
        "steps": [
            ("Explain correlation vs causation with an example.",
             "Correlation means two variables move together. Causation means one directly affects the other. Correlation does not imply causation."),
            ("Example: ice cream sales and drowning are correlated (both rise in summer), but ice cream does not cause drowning. The confounder is hot weather.",
             "This is why randomized controlled trials are the gold standard for establishing causation."),
        ],
    },
    {
        "title": "The Sunk Cost Fallacy",
        "steps": [
            ("What is the sunk cost fallacy?",
             "The sunk cost fallacy is the tendency to continue investing in a losing endeavor because of past unrecoverable investments (time, money, effort) rather than cutting losses."),
            ("Rational decisions should ignore sunk costs and only consider future costs and benefits. Example: continuing a bad movie because you already paid for the ticket.",
             "The fallacy is driven by loss aversion and the desire to avoid admitting past mistakes."),
        ],
    },
    {
        "title": "Anchoring Bias",
        "steps": [
            ("Define anchoring bias and give an example.",
             "Anchoring bias is relying too heavily on the first piece of information encountered (the anchor) when making decisions. Subsequent judgments are adjusted from that anchor."),
            ("Example: in salary negotiations, the first offer sets an anchor. A higher initial anchor leads to higher final settlements, even if the anchor is arbitrary.",
             "This is robust even when people know the anchor is irrelevant."),
        ],
    },
    {
        "title": "Tversky and Kahneman's Linda Problem",
        "steps": [
            ("Linda: 31, single, outspoken, bright, philosophy major, concerned with social justice. Which is more probable: (1) Linda is a bank teller, or (2) Linda is a bank teller and active in the feminist movement?",
             "The conjunction fallacy: P(A and B) cannot exceed P(A). Option 2 is a subset of option 1. Yet most people choose option 2 because it is more representative."),
            ("This demonstrates how intuitive reasoning violates probability laws. Kahneman won the Nobel Prize in Economics partly for work on these heuristics and biases.",
             "The conjunction fallacy persists even when explicitly explained."),
        ],
    },
    {
        "title": "The Framing Effect",
        "steps": [
            ("Explain the framing effect in medical decisions.",
             "People react differently to identical information depending on presentation. Surgery with 90% survival rate is chosen more than surgery with 10% mortality rate, though they are the same."),
            ("The framing effect shows how emotions influence rational decision-making. Loss frames (mentioning what could be lost) tend to be more powerful than gain frames.",
             "This is exploited in marketing and political messaging."),
        ],
    },
]


def expand_thinking(rng, step_q: str, step_a: str) -> list[tuple[str, str]]:
    """
    Given a Q/A pair from a topic, produce one long thinking block and one clean
    response block.
    """
    templates = [
        "Okay, let me think this through carefully.",
        "Right, let me work through this step by step.",
        "Alright, I need to reason about this systematically.",
        "Let me tackle this one piece at a time.",
    ]
    bridges = [
        "From here, the next step is clear:",
        "Now I can put this together:",
        "This tells me that:",
        "So working through the details:",
    ]

    thought_parts = [
        f"{rng.choice(templates)} {step_q}",
        step_a,
        f"{rng.choice(bridges)} {step_a}",
    ]

    # Add a self-check or meta thought to extend the monologue
    if rng.random() < 0.5:
        checks = [
            "Let me double-check that reasoning. If I work through it again from the top, I want to make sure I have not missed any subtlety or made an unwarranted assumption. Going through each piece once more confirms the logic holds up.",
            "I should verify this against what I know. The relationships involved are well-established, so if my setup is correct the conclusion should follow. Let me trace through one more time to be safe.",
            "Hmm, does this actually make sense? Let me try a quick mental sanity check using different numbers or a special case to see if the result is plausible.",
            "Before I commit to this answer, let me consider whether there is another way to arrive at the same result. If a different method gives the same answer, I can be much more confident.",
        ]
        thought_parts.insert(2, rng.choice(checks))

    full_thought = " ".join(thought_parts)

    response_templates = [
        "The answer is: {ans}",
        "{ans}",
        "So {ans}",
        "Therefore, {ans}",
    ]
    response = rng.choice(response_templates).format(ans=step_a.rstrip("."))

    return [(full_thought, response)]


def format_file(title: str, thought_pairs: list[tuple[str, str]], rng) -> str:
    """Format a file with alternating thinking and par blocks."""
    lines = [f"<|title|>{title}<|/title|>", ""]
    for thought_text, response_text in thought_pairs:
        lines.append(f"<|thinking|>{thought_text}<|/thinking|>")
        lines.append("")
        lines.append(f"<|par|>{response_text}<|/par|>")
        lines.append("")
    lines.append("<|endoftext|>")
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
        nums = [int(re.match(r"synthetic(\d+)\.txt", f).group(1)) for f in existing if re.match(r"synthetic(\d+)\.txt", f)]
        start = max(nums) + 1 if nums else 1
    else:
        start = 1

    for i in range(args.count):
        topic = rng.choice(TOPICS)
        idx = start + i
        all_pairs = []
        for q, a in topic["steps"]:
            all_pairs.extend(expand_thinking(rng, q, a))
        content = format_file(topic["title"], all_pairs, rng)
        fname = f"synthetic{idx:04d}.txt"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {args.count} files -> {args.out}")


if __name__ == "__main__":
    main()
