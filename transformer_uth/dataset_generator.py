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
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Callable, List

from num2words import num2words

_LETTERS = "ABCD"
_MAX_LENGTH = 9

_PATH_DATA = Path(__file__).parent / "data"


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


def _generate_random_sequences() -> List[str]:
    random.seed(324)
    return list({_input_text() for _ in range(2_000_000)})


def _generate_output(
    input_sequences: List[str], func_task: Callable[[str], str]
) -> List[str]:
    return [func_task(input_seq) for input_seq in input_sequences]


def _generate_dataset(
    path_dataset: Path, input_sequences: List[str], output_sequences: List[str]
):
    raw_text = ""
    for input_seq, output_seq in zip(input_sequences, output_sequences):
        row = "{0}\t{1}\n".format(input_seq, output_seq)
        raw_text += row

    path_dataset.write_text(raw_text)
    logging.warning("Dataset stored in %s", path_dataset)


def _generate_task_datasets():
    _PATH_DATA.mkdir(exist_ok=True, parents=True)

    tasks = (
        ("task1-data.tsv", _output_task1),
        ("task2-data.tsv", _output_task2),
        ("task3-data.tsv", _output_task3),
        ("task4-data.tsv", _output_task4),
    )
    random_inputs = _generate_random_sequences()
    for task in tasks:
        path_dataset = _PATH_DATA / task[0]
        if not path_dataset.exists():
            outputs = _generate_output(random_inputs, task[1])
            _generate_dataset(path_dataset, random_inputs, outputs)


def _generate_number_translation_datasets():
    num_numbers = 1_000_000
    es_numbers = [
        num2words(number, lang="es") for number in range(num_numbers)
    ]
    en_numbers = [
        num2words(number, lang="en") for number in range(num_numbers)
    ]
    output_numbers = [str(number) for number in range(num_numbers)]

    tasks = (
        ("es_numbers_translation.tsv", es_numbers),
        ("en_numbers_translation.tsv", en_numbers),
    )
    for task in tasks:
        path_dataset = _PATH_DATA / task[0]
        if not path_dataset.exists():
            _generate_dataset(path_dataset, task[1], output_numbers)


if __name__ == "__main__":
    _generate_task_datasets()
    _generate_number_translation_datasets()
