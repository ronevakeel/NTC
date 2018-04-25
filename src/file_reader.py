"""
    A simple reader which reads plain noisy text.

    Author: Haobo Gu, Yuanhe Tian
    Date created: 04/15/2018
    Python version: 3.6.2
"""

import operator
import nltk

data_path = '../data/'

def read_OCR_file(filename):
    """
        Read all contents in file
        :return: list of strings, one string per line
    """
    import codecs
    with codecs.open(data_path + filename, "r", encoding='utf-8', errors='ignore') as input_file:
        contents = input_file.read()
        contents = contents.split('\r\n')
    return contents


def clean_empty_line(raw_lines):
    raw_content_no_empty_lines = []
    for raw_line in raw_lines:
        raw_line = raw_line.strip()
        if operator.eq(raw_line, ""):
            continue
        else:
            raw_content_no_empty_lines.append(raw_line)
    return raw_content_no_empty_lines


def lines2string(lines):
    content_string = ""
    for line in lines:
        content_string += line + " "
    content_string.strip()
    return content_string


# Test script

if __name__ == "__main__":
    raw_content = read_OCR_file('newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt')
    content =  clean_empty_line(raw_content)
    # print(content)
    '''
    for item in content:
        tokens = nltk.tokenize.word_tokenize(item)
        print(item)
        print(tokens)
        print("")
    '''
    clean_string = lines2string(content)
    print("hello")