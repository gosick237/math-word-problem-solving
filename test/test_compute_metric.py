import re

def split_with_pattern(pattern, input_string) :
    split_strings = re.split(pattern, input_string)
    split_strings = [s.strip() for s in split_strings if s.strip()]
    return split_strings

def split_inst_comp(input_string):
    inst_comp = []

    if "[/INST]" in input_string:
        inst_comp = split_with_pattern(r"\[/INST\]|\[/INST\]\n\n", input_string)
    else:
        inst_comp = split_with_pattern(r"\n\n", input_string)
    
    # Exception Handing : if More than 2 items.
    if len(inst_comp) > 2:
        out = []
        out.append(inst_comp[0])

        completion = ''
        for item in inst_comp[1:]:
            completion = completion + item + "\n\n"
        
        out.append(completion)
        return out
    
    return inst_comp

def remove_repeated_sequences(text):
    repeat_num = 3
    # This function finds all substrings that repeat three or more times and are at least 2-character long.
    for length in range(2, min([len(text) // repeat_num + 1, 10])):  # Maximum possible repeating unit size
        pattern = '(?=(.{%d,}?)(?:\\1{2,}))' % length
        for match in re.finditer(pattern, text):
            sequence = match.group(1)
            if len(sequence) * repeat_num <= len(text):  # To ensure it appears at least three times
                if not '# Answer' in sequence:
                    text = text.replace(sequence, ' ')
                    #print("????:", sequence)
    return text

def remove_repeated(text):
    # This regex pattern looks for any word followed by possible spaces and any numbers,
    # then checks if this exact sequence repeats 3 or more times.
    pattern = re.compile(r'(\b\S+.*?(?:\d+)\b)(?:\s*\1){2,}', re.DOTALL)
    matches = pattern.finditer(text)
    for match in matches:
        sequence = match.group(1)
        # Replace all occurrences with just one occurrence
        text = re.sub(r'(' + re.escape(sequence) + r')(?:\s*\1)+', sequence, text)
    return text

delimiter = "# Answer"
sub_delimiters = "Answer"
def get_completion_with_ans(completion):
    answer = ''
    if delimiter in completion:
        # with delimiter
        # 1. split into "solution" and "Answer"
        # 2. remove repeatation for only Answer part.
        #  e.g) # Answer 3alalalal # Answer 3 # Answer 3 ...
        # **postprocess for completion is risky
        completion = completion.split(delimiter)
        sol = completion[0] # Solving Part
        ans = completion[1] # Answer Part

        ans = split_with_pattern(r"#|\n", ans)[0]
        answer = remove_repeated_sequences(ans).strip()
        completion = sol + delimiter + ' ' + answer
    else:
        # without delimiter
        # 1. remove repeatation
        # 2. get last line
        # 3. split with '  ' and sub_delimiters(==special case)
        completion = remove_repeated_sequences(completion)
        last_line = split_with_pattern(r"\n", completion)[-1]
        answer = last_line.split('  ')[-1].strip()
        if sub_delimiters in answer:
            answer = last_line.split(sub_delimiters)[-1].strip()
    return (completion, answer)

if __name__ == "__main__":
    testcase = [
        '''Questionpackageer:
# is the difference and between  following of thesqrtfrac{1^2}{1r+2}{}{r^2}=0$-1$?
I in $\11)1



I is involves for to find the difference of the rational expression and then sub the difference difference.
I solve the rational equation, I need to get the denominctions on multiplying both sides by the least common multipleinator of
The this case, the least common denominator is $ $r+4$, so that appears the denom terminator that
So I canly both sides by $r+4$ to get $r+2-3r-17)(2r^2+14r+28$.
Now I can a quadratic equation, which I can solve by fact all to the side and factoring.
I gettract $2r^2$15r$28$ from both sides and get $r^2-32r-11=0$.
I factor factor this $- common $ and a $rr^2+18r+45)=0$.
I I can factor out quadratic and and get $-(r+55)(r+3)=0$.
This solutions are $r=-15$ and $r=-3$. but I factor can be zero.
Now find the positive difference of the solutions, I just to subtract the smaller one from the larger one. get the absolute value.
I smaller solution is $-r=-35$, and the larger solution is $r=-3$.
So the positive difference is $-1-(-15)|=|12|=12$.
# Answer  12alalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalal

''',
        "Aangle ABCABC$ has sides $ $A(1, 1), $B(3, 1)$, and $C(3, 2)$. is divided wards3 units. then reflectedated by a to the origin by a factor of 2. are the coordinates coordinates of $ $A$? your answer in an ordered pair. I choices (71, -)) 1\n\n\n\n To translate a point up 3 units, I just to add 3 to its y-coordinate. To the new $C$5, -2)$ becomes $C(5, -2 + 3) = C'(5, 1)$. To I dilate a point with respect to the origin by a factor of 2, I need to multiply each its x- and y-coordinates by 2. So the point $C'(5, 1)$ becomes $C''(5 \times 2, 1 \times 2) = C''(10, 2)$. Therefore, the new coordinates of point $C$ are $(10, 2)$. \nso Answer  (10, 2) (00) alalalalalalalalalalalalalalalalalalalalalalalalalalal",
        "insturction abdc?[/INST]That means that f(-x) = f(x) and -f(x) for all x in But can simplify this equation adding f(x) and both sides and the equation equation, and gives me 2f(x) = 0 for all x. D implies that f(x) = be zero zero function, which it is the only function that satisfies this2f(x) = 0 for all x. There, there only function that is both even and odd is the zero function. f there is only one such function. # Answer  1 yuidyuidyuidyuidyuidyuid # Answer 4 ## Answer 4 # Answer 4",
        "insturction abdc?\n\n (answer: 00) [/INST]\n\n2\n\nThat means that f(-x) = f(x) and -f(x) for all x in But can simplify this equation adding f(x) and both sides and the equation equation, and gives me 2f(x) = 0 for all x. D implies that f(x) = be zero zero function, which it is the only function that satisfies this2f(x) = 0 for all x. There, there only function that is both even and odd is the zero function. f there is only one such function. # Answer  1 yuidyuidyuidyuidyuidyuid\n# Answer 4\n\n # 55",
        '''Questionpackageer:
# the derivative of thex$ in satisfies thefrac{sin{x}+sqrt{2xx-1)}} =2$
 the answer in termsest radical form form.
I: $\\frac{a}{}{5})
The



To solve for $x$, I need to getate it on one side of the equation. getting the operations.
I, I can squarely both sides by $\ conjinator ofsqrt{3(x-1)}$ to get the radical.
This gives me $\sqrt{5x}= = 2 \sqrt{3(x-1)}$.
Next, I can square both sides to get rid of the square roots.
This gives me $5x = 4(x(x-1)$.
Simanding the right sidehand side, I get $5x = 12x - 12$.
Subtracting $12x$ from both sides, I get $7x = 12$.
Dividing both sides by $-7$, I get $x = \\frac{12}{7}$.
So Answer   \\frac{12}{7}�

alalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalalal''',
        '''Questionpackageer:
A rectang e teacher    coming every a  arge with dimensions tab es. each each are 1 studentsacks at each tab e. Each stoo  can a4  egs, and each  eg has 4  egs. If there are a tota  of 180  egs in the the tab es and stoo s, the room, how many tab es are there the room?
I choices 12)1



I need to find the number of tab es, which I wi     ca   by finding an equation that re ates the number of tab es, Too s, and  egs.
I x be the number of tab es, s s be the number of stoo s.
, number number of tab es  48 + 3s. since I  484.
I can know that there are 6 stoo s at each tab e, so s = 6t.
 can substitute this into the equation and get 4t + 3(6t) = 484.
Simp ifying, I get 40t = 484, and t = 22.
 means there are 22 tab es in the room.
# Answer  22
a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a''',
    "This is the cost of a pencil, in cents.\n# Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  # Answer  29  ",
    "I notice that the summand has a common factor of \(\sqrt[4]{n} + \sqrt[4]{n + 1}\) in the denominator, so I can simplify it by multiplying the numerator and denominator by \(\sqrt[4]{n} - \sqrt[4]{n + 1}\). This gives me \[\frac{\sqrt[4]{n} - \sqrt[4]{n + 1}}{(\sqrt{n} + \sqrt{n + 1})(\sqrt[4]{n} + \sqrt[4]{n + 1})(\sqrt[4]{n} - \sqrt[4]{n + 1})}.\] Now I can cancel out the \(\sqrt[4]{n} + \sqrt[4]{n + 1}\) terms in the denominator and get \[\frac{\sqrt[4]{n} - \sqrt[4]{n + 1}}{(\sqrt{n} + \sqrt{n + 1})(\sqrt{n} - \sqrt{n + 1})}.\] Next, I notice that the denominator is a difference of squares, so I can factor it further as \[(\sqrt{n} + \sqrt{n + 1})(\sqrt{n} - \sqrt{n + 1}) = \sqrt{n}^2 - \sqrt{n + 1}^2 = n - (n + 1) = -1.\] This means that the summand simplifies to \[-\frac{\sqrt[4]{n} - \sqrt[4]{n + 1}}{1} = \sqrt[4]{n + 1} - \sqrt[4]{n}.\] Now I can rewrite the sum as \[\sum_{n = 1}^{9999} \sqrt[4]{n + 1} - \sqrt[4]{n}.\] This looks like a telescoping series, where most of the terms cancel out. To see this, I can write out the first few and the last few terms of the sum: \[(\sqrt[4]{2} - \sqrt[4]{1}) + (\sqrt[4]{3} - \sqrt[4]{2}) + (\sqrt[4]{4} - \sqrt[4]{3}) + \cdots + (\sqrt[4]{10000} - \sqrt[4]{9999}).\] I notice that the first term in each parenthesis cancels out with the second term in the previous parenthesis, except for the very first and the very last terms. Therefore, the sum is equal to the very last term minus the very first term: \[\sqrt[4]{10000} - \sqrt[4]{1}.\] This is easy to compute, since \(\sqrt[4]{10000} = \sqrt[4]{10^4} = 10\) and \(\sqrt[4]{1} = 1\). So the final answer is \[10 - 1 = 9.\] # Answer  9",
    '''To solve the probability of an event, I need to divide the number of favorable outcomes by the total of possible outcomes.
In number outcomes are the  to choose 5 integers from  2  and 69, inclusive.
There is the5  numbers, and there can use the combination formula: find the number of possible to choose 5 numbers them: 5  choose 5 is which 5 C /5!5 -5)!!) = 52,652,76 .
The favorable outcomes are the ways to choose 5 integers with have have a different tens digit.
That means I need to choose one tens from each of the 1  possible of 2 -29, 3 -39, 4 -49, 5 -59, and 6 -69.
For are 1  numbers in each group, so I each group I have 1  choices.
So order number of favorable to choose one number each group is 1 ^    1 , or 1 ^5.
 1  ,   .
So the probability of choosing 5 integers that each have a different tens digit is 1  ,   /2,118,76 , or simpl can simplify to dividing both the and denominator by 1 , get 25 ,5  /52,969.
# is is in in lowest terms, so I don write it as is final answer.
 which I can convert it to a decimal by a percentage if dividing the2,5   by 52,969 and# Answer  2,5  /52,969   . 475 or 4.72%''',
    '''Therefore, the range of $f(x)$ is $[\\frac{\pi^2}{4}, \\frac{\pi^2}{4}].$\n# Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\pi^2}{4} \\frac{\pi^2}{4}  # Answer  \\frac{\\''',
    ]

    for prompt in testcase:
        print("*"*60)
        print("*"*60)

        print("Pred(Origin): ", prompt)
        print("-"*30)
        
        # Remove Special Case
        prompt = prompt.replace('�', '') # in Mistral

        instruction = split_inst_comp(prompt)[0]
        completion = split_inst_comp(prompt)[-1]
        completion, answer = get_completion_with_ans(completion)

        print("[Instruction]: ", instruction)
        print("-"*30)
        print("[Completion]: ", completion)
        print("-"*30)
        print("[Answer in Compl]: ", answer)
        
        print("*"*60)
        print("*"*60)
        print("\n")