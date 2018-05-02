import operator
import re
import os
import nltk

data_path = "../data/local/"
model_path = "../output/"
WHITE_SPACE = 0
TOKENIZER = 1


def readfile(file_name):
    '''
    Get a list of lines from the file.
    :param file_name: path of input file
    :return: a list of lines in this file
    '''
    content = []
    lines = open(file_name, 'r').readlines()
    for line in lines:
        line = line.strip()
        ls = re.split("<[^<>]*>", line)
        line = ""
        for l in ls:
            line += l
        if operator.eq(line, "") or re.match("\\s+", line) or re.match("<[^>]*", line) or re.match("[^<]*>", line):
            continue
        content.append(line)
    return content


def writefile(unigram, bigram, path=model_path):
    '''
    Write the count of all unigrams and bigrams into different file
    :param unigram: a dictionary of count of unigram
    :param bigram: a dictionary of count of bigram
    :param path: the path to store the two output file
    '''
    uni_file = open(os.path.join(path, "unigram"), 'w')
    bi_file = open(os.path.join(path, "bigram"), 'w')

    for item, value in unigram.items():
        uni_file.write(item + " " + str(value) + "\n")
    for item, value in bigram.items():
        bi_file.write(item[0] + " " + item[1] + " " + str(value) + "\n")


def get_files(dir, file_list):
    '''
    From the data directory, get a list of all readable files' path
    :param dir: the directory of data
    :param file_list: a list to store the path of files
    '''
    files = os.listdir(dir)
    for file in files:
        if operator.eq(file, ".DS_Store"):
            continue
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            file_list.append(path)
        elif os.path.isdir(path):
            get_files(path, file_list)


def count_appearance(content, ugram_dict, bgram_dict, split_strategy=TOKENIZER):
    '''
    Given a list of lines, count the appearance of unigram and bigrams in a specific tokenizing strategy.
    :param content: list of lines
    :param ugram_dict: the dictionary that stores the count of all unigrams
    :param bgram_dict: the dictionary that stores the count of all bigrams
    :param split_strategy: the way to split words, WHITE_SPACE or NLTK tokenizer
    '''
    for line in content:
        items = []
        if split_strategy == WHITE_SPACE:
            ''' Use white space to split lines'''
            items = re.split("\\s+", line)
        elif split_strategy == TOKENIZER:
            ''' Use nltk tokenizer to split lines'''
            items = nltk.tokenize.word_tokenize(line)

        for item in items:
            ''' Count unigrams '''
            if item in ugram_dict:
                ugram_dict[item] += 1
            else:
                ugram_dict[item] = 1

        for i in range(1, len(items)):
            ''' Count bigrams '''
            bg = (items[i-1], items[i])
            if bg in bgram_dict:
                bgram_dict[bg] += 1
            else:
                bgram_dict[bg] = 1


def ngrammodel(data_path, output_path=model_path):
    '''
    Given a data directory and output directory, generate files that store the count of unigrams and bigrams
    :param data_path: the directory of training data
    :param output_path: the directory to store counting data
    '''
    all_files = []
    unigramdict = {}
    bigramdict = {}
    get_files(data_path, all_files)
    for file in all_files:
        try:
            contentlist = readfile(file)
        except Exception:
            print(file)
            continue
        count_appearance(contentlist, unigramdict, bigramdict, TOKENIZER)
    writefile(unigramdict, bigramdict, output_path)


def wf_levenshtein(string_1, string_2):
    """
    Calculates the Levenshtein distance between two strings.

    This version uses the Wagner-Fischer algorithm.

    Usage::

        >>> wf_levenshtein('kitten', 'sitting')
        3
        >>> wf_levenshtein('kitten', 'kitten')
        0
        >>> wf_levenshtein('', '')
        0

    """
    len_1 = len(string_1) + 1
    len_2 = len(string_2) + 1

    d = [0] * (len_1 * len_2)

    for i in range(len_1):
        d[i] = i
    for j in range(len_2):
        d[j * len_1] = j

    for j in range(1, len_2):
        for i in range(1, len_1):
            if string_1[i - 1] == string_2[j - 1]:
                d[i + j * len_1] = d[i - 1 + (j - 1) * len_1]
            else:
                d[i + j * len_1] = min(
                   d[i - 1 + j * len_1] + 1,        # deletion
                   d[i + (j - 1) * len_1] + 1,      # insertion
                   d[i - 1 + (j - 1) * len_1] + 1,  # substitution
                )

    return d[-1]


if __name__ == "__main__":
    ngrammodel(data_path)
