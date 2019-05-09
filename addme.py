from __future__ import print_function  # for compatibility with Python 2
import string

# Insert the metapuzzle answer here
meta_answer = "GRA_EDAY"



# Insert your puzzle answers here
puzzles = {
    "07-131": "GITBRANCH",
"15-151": "DIETCOKE",
"15-112": "CAMPOUTINOFFICEHOURS",
"15-122 A": "MASTERPIECE",
"15-122 B": "LUMOS",
"15-150 A": "VALUES",
"15-150 B": "TIED UP",
"15-251": "GITBRANCH",
"15-213": "KILLSIGNAL"
}

# Format the answers to have no spaces and be in all caps
def format_answer(answer):
    return answer.upper().replace(" ", "")


meta_answer = format_answer(meta_answer)
for course in puzzles:
    puzzles[course] = format_answer(puzzles[course])


# ==========HELPER FUNCTIONS==========
# Takes in a character and outputs the character's order in the alphabet
# e.g., char2num("A") == 1 and char2num("Z") == 26
def char2num(char):
    num = ord(char) - 64
    return num


# Takes in a number and outputs the number-th character in the alphabet
# e.g., num2char(1) == "A" and num2char(26) == "Z"
def num2char(num):
    char = chr(num + 64)
    return char


# ==========TESTS BEGIN HERE==========
# The indices for Python lists begin at zero!!!
def test0(letter):
    if letter == puzzles["15-213"][6]:
        return True
    else:
        return False


def test1(letter):
    if letter == puzzles["15-122 A"][5]:
        return True
    else:
        return False


def test2(letter):
    string = puzzles["15-112"]
    num = 0
    for i in range(len(string)):
        if i % 2 == 0:
            num -= char2num(string[i])
        else:
            num += char2num(string[i])

    if letter == num2char(num):
        return True
    else:
        return False


def test3(letter):
    set1 = set(puzzles["07-131"])
    set2 = set(puzzles["15-251"])
    intersection = set1 & set2
    print(sorted(intersection)[1])

    for i in string.ascii_letters:
        if letter == i:
            print(i)



    if letter == sorted(intersection)[1]:
        return True
    else:
        return False


def test4(letter):
    num = 0
    for char in puzzles["15-122 B"]:
        num += char2num(char)
    for char in puzzles["15-150 B"]:
        num -= char2num(char)

    if letter == num2char(num):
        return True
    else:
        return False


def test5(letter):
    string = puzzles["15-151"]

    if letter == sorted(string)[1]:
        return True
    else:
        return False


def test6(letter):
    return test2(letter)


def test7(letter):
    func_str = puzzles["15-150 A"].lower()
    lst = eval("puzzles." + func_str + "()")
    lens = list(map(len, lst))
    print(num2char(min(lens) + max(lens)))

    if letter == num2char(min(lens) + max(lens)):
        return True
    else:
        return False


def validate_meta_answer(meta_answer, puzzles):
    if len(meta_answer) != 8:
        print("****FAILED****")

    tests = [test0, test1, test2, test3, test4, test5, test6, test7]
    for i in range(8):
        print("Running test " + str(i) + "...")
        if tests[i](meta_answer[i]):
            print("****PASSED****")
        else:
            print("****FAILED****")


validate_meta_answer(meta_answer, puzzles)