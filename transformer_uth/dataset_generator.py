"""
Script to generate different task datasets:

Some dataset inspired on https://github.com/greentfrapp/attention-primer

task1:
Counting letters
"ABCDABC" -> "2221"

task2:
Counting letters + Difference between A & B
"AABA"    -> "31002"

task3:
Signal
"ABBBAA" -> "2"

task4:
Multiple Signals (3 signals)
"ABBBAA" -> "211"
"""
import random
from collections import Counter

_LETTERS = "ABCD"
_MAX_LENGTH = 9


def _input_text() -> str:
    return "".join(random.choices(_LETTERS, k=random.randint(4, _MAX_LENGTH)))


def _output_task1(input_text: str) -> str:
    """Task 1 is count letters.

    * (numbers for each letter are always in the same position)

    example:
    input  -> "BBCCCCCCD"
    output -> "0261"
    """
    counter_letters = Counter(input_text)
    return "".join(str(counter_letters.get(letter, 0)) for letter in _LETTERS)


def _output_task2(input_text: str) -> str:
    """Task 2 is count letters adding absolute number difference between A & B.

    * last digit is absolute difference between numbers of A & numbers of B

    example:
    input  -> "ABA"
    output -> "21001"
    """
    count_letters = _output_task1(input_text)
    a, b, *_ = [int(i) for i in count_letters]
    difference = str(abs(a - b))
    return count_letters + difference


def _output_task3(input_text: str) -> str:
    """Task 3 is count only signal letter.

    * The signal is the first letter,
    * the output number is how many times appears the signal letter

    example:
    input  -> "BBAA"
    output -> "1"

    "B" appears only once
    """
    signal_char = input_text[0]
    return str(input_text[1:].count(signal_char))


def _output_task4(input_text: str) -> str:
    """Task 4 is count only signal letters.

    * Is the same as task 3 but using more signals

    example:
    input  -> "ABBBAA"
    output -> "211"

    "A" appears 2 times
    "B" apperas 1 time
    """
    return "".join(str(input_text[3:].count(c)) for c in input_text[:3])
