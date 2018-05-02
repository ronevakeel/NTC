import operator
import re
import os
import nltk

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


def writefile(unigram, bigram, path):
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


def count_appearance(content, ugram_dict, bgram_dict, total_token, split_strategy=TOKENIZER):
    '''
    Given a list of lines, count the appearance of unigram and bigrams in a specific tokenizing strategy.
    :param content: list of lines
    :param ugram_dict: the dictionary that stores the count of all unigrams
    :param bgram_dict: the dictionary that stores the count of all bigrams
    :param split_strategy: the way to split words, WHITE_SPACE or NLTK tokenizer
    '''
    for line in content:
        items = splitstr(line, split_strategy)
        for item in items:
            ''' Count unigrams '''
            if item in ugram_dict:
                ugram_dict[item] += 1
            else:
                ugram_dict[item] = 1
            total_token += 1

        for i in range(1, len(items)):
            ''' Count bigrams '''
            bg = (items[i-1], items[i])
            if bg in bgram_dict:
                bgram_dict[bg] += 1
            else:
                bgram_dict[bg] = 1


def splitstr(line, split_strategy):
    '''
    Split a string by specific strategy, white space is the default.
    :param line: the string to be splited
    :param split_strategy: a way to split the string
    :return: a list of fragments
    '''
    items = []
    if split_strategy == TOKENIZER:
        ''' Use nltk tokenizer to split lines'''
        items = nltk.tokenize.word_tokenize(line)
    else:
        ''' Use white space to split lines'''
        items = re.split("\\s+", line)
    return items


def ngrammodel(data_path, output_path):
    '''
    Given a data directory and output directory, generate files that store the count of unigrams and bigrams
    :param data_path: the directory of training data
    :param output_path: the directory to store counting data
    '''
    all_files = []
    unigramdict = {}
    bigramdict = {}
    total_tokens = 0
    get_files(data_path, all_files)
    for file in all_files:
        try:
            contentlist = readfile(file)
        except Exception:
            print(file)
            continue
        count_appearance(contentlist, unigramdict, bigramdict, total_tokens, TOKENIZER)
    writefile(unigramdict, bigramdict, output_path)
    return unigramdict, bigramdict, total_tokens


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


def modify_line(unigram, bigram, input_line, total_tokens, split_strategy, topN, delta=1):
    words = splitstr(input_line, split_strategy)
    line_candidate = []
    for word in words:
        if word in unigram:
            line_candidate.append([word])
        else:
            candidates = get_candidate(word, unigram, topN)
            line_candidate.append(candidates)
    return bestoutput(unigram, bigram, line_candidate, total_tokens, delta)


def bestoutput(unigram, bigram, line_candidates, total_tokens, delta):
    sen_length = len(line_candidates)
    begin_candidates = line_candidates[0]
    from_state = []
    for cand in begin_candidates:
        from_state.append(("", get_unigram_prob(cand, unigram, delta, total_tokens)))
    for i in range(1, sen_length):
        prev_candidates = line_candidates[i-1]
        curr_candidates = line_candidates[i]
        for j in len(curr_candidates):
            curr_cand = curr_candidates[j]
            for k in len(prev_candidates):
                prev_cand = prev_candidates[k]
                bg = (prev_cand, curr_cand)


def get_unigram_prob(unigram, unigram_dict, delta, total_takens):
    voc_size = len(unigram_dict)
    cw = delta
    if unigram in unigram_dict:
        cw += unigram_dict[unigram]
    total = delta * voc_size + total_takens
    return float(cw) / total


def get_bigram_prob(bigram, unigram_dict, bigram_dict, delta):
    first = bigram[0]
    second = bigram[1]
    voc_size = len(unigram_dict)
    cw = delta
    if bigram in bigram_dict:
        cw += bigram_dict[bigram]
    total = delta *


def get_candidate(err_word, unigram_dic, topN):
    candidate = []
    if topN > len(unigram_dic):
        return list(unigram_dic.keys())
    else:
        for word in unigram_dic.keys():
            cost = wf_levenshtein(err_word, word)
            if len(candidate) < topN:
                candidate.append((word, cost))
            else:
                biggest_index, biggest_cost = find_biggest(candidate)
                if cost < biggest_cost:
                    candidate[biggest_index] = (word, cost)
    return candidate

def find_biggest(list):
    index = 0
    cost = -1
    for i in range(0, len(list)):
        curr_cost = list[i][1]
        if curr_cost > cost:
            cost = curr_cost
            index = i
    return index, cost


if __name__ == "__main__":
    data_path = "../data/local/"
    model_path = "../output/"
    unigram, bigram, total_tokens = ngrammodel(data_path, model_path)

    text_path = "../data/OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt"
    train_file = open(text_path, 'r')
    for line in train_file.readlines():
        modify_line(unigram, bigram, line, total_tokens, TOKENIZER, 10, 1)

