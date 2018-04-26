"""
    A simple reader which reads plain noisy text.

    Author: Haobo Gu, Yuanhe Tian
    Date created: 04/15/2018
    Python version: 3.6.2
"""

import operator
import re
import nltk

data_path = '../data/'

def read_file(filename):
    """
    Read all contents in file
    :return: list of strings, one string per line
    """
    import codecs
    with codecs.open(data_path + filename, "r", encoding='utf-8', errors='ignore') as input_file:
        contents = input_file.read()
        contents = contents.split('\n')
    return contents


def clean_empty_line(raw_lines):
    no_empty_lines = []
    for raw_line in raw_lines:
        raw_line = raw_line.strip()
        raw_line = re.sub("\\s+", " ", raw_line)
        if operator.eq(raw_line, ""):
            continue
        else:
            no_empty_lines.append(raw_line)
    return no_empty_lines


def lines2string(lines):
    content_string = ""
    for line in lines:
        content_string += line + " "
    content_string.strip()
    return content_string


def get_pairs(ocr_file, gold_standard):
    raw_content = read_file('OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt')
    gold_content = read_file('gold_standard/Edit_ Some Further Accounts of the Nile 1912-1913.txt')
    r_content = clean_empty_line(raw_content)
    g_content = clean_empty_line(gold_content)

    return (lines2string(r_content), lines2string(g_content))

# Test script

if __name__ == "__main__":
    raw_content = read_file('OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt')
    gold_content = read_file('gold_standard/Edit_ Some Further Accounts of the Nile 1912-1913.txt')
    r_content = clean_empty_line(raw_content)
    g_content = clean_empty_line(gold_content)
    # print(content)
    '''
    for item in content:
        tokens = nltk.tokenize.word_tokenize(item)
        print(item)
        print(tokens)
        print("")
    '''
    r_clean_string = lines2string(r_content)
    g_clean_string = lines2string(g_content)
    print("hello")