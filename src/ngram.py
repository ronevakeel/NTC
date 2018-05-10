import operator
import re
import os
import nltk
import math

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


def count_appearance(content, ugram_dict, bgram_dict, split_strategy=TOKENIZER):
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
            item = item.lower()
            if item in ugram_dict:
                ugram_dict[item] += 1
            else:
                ugram_dict[item] = 1

        for i in range(1, len(items)):
            ''' Count bigrams '''
            bg = (items[i-1].lower(), items[i].lower())
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
    get_files(data_path, all_files)
    for file in all_files:
        try:
            contentlist = readfile(file)
        except Exception:
            # print(file)
            continue
        count_appearance(contentlist, unigramdict, bigramdict, TOKENIZER)
    writefile(unigramdict, bigramdict, output_path)
    total_tokens = sum(unigramdict.values())
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

    d = [0.0] * (len_1 * len_2)

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
                   d[i - 1 + (j - 1) * len_1] + 0.8,  # substitution
                )

    return d[-1]


def modify_line(unigram, bigram, input_line, total_tokens, split_strategy, topN, delta, threshold):
    '''
    Get the correction of a given line of string
    :param unigram: a dictionary of unigrams in the training corpus
    :param bigram: a dictionary of bigrams in the training corpus
    :param input_line: the line of string waiting to be corrected
    :param total_tokens: the sive of the training corpus
    :param split_strategy: the way to split tokens
    :param topN: the number of candidates for each word waiting to be corrected
    :param delta: the value of delta used to smooth the probability
    :param threshold: the threshold for the cost of candidate selection
    :return: a corrected line of string with highest generation probability
    '''
    words = splitstr(input_line, split_strategy)
    line_candidate = []
    for word in words:
        if word in unigram:
            line_candidate.append([(word, 1)])
        else:
            candidates = get_candidate(word, unigram, topN, threshold)
            line_candidate.append(candidates)
    corrected_words = bestoutput(unigram, bigram, line_candidate, total_tokens, delta)

    return replace_words(input_line, words, corrected_words)


def bestoutput(unigram, bigram, line_candidates, total_tokens, delta):
    """
    Use Viterbi algorithm to get the best output.
    :param unigram:
    :param bigram:
    :param line_candidates:
    :param total_tokens:
    :param delta:
    :return:
    """
    sen_length = len(line_candidates)
    begin_candidates = line_candidates[0]
    from_state = []
    for cand in begin_candidates:
        from_state.append(([], math.log(get_unigram_prob(cand, unigram, delta, total_tokens))))
    for i in range(1, sen_length):
        prev_candidates = line_candidates[i-1]
        curr_candidates = line_candidates[i]
        to_state = []
        for j in range(len(curr_candidates)):
            curr_cand = curr_candidates[j]
            prob_list = []
            for k in range(len(prev_candidates)):
                prev_cand = prev_candidates[k]
                bg = (prev_cand, curr_cand)
                bi_prob = get_bigram_prob(bg, unigram, bigram, delta, total_tokens)
                path = from_state[k][0].copy()
                path.append(k)
                prob_list.append((path, from_state[k][1] + math.log(bi_prob)))
            best_index = get_best_candidate(prob_list)
            to_state.append(prob_list[best_index])
        from_state = to_state.copy()
    best_index = get_best_candidate(from_state)
    from_state[best_index][0].append(best_index)
    best_list = from_state[best_index][0]
    best_candidate = []
    for i in range(len(best_list)):
        candidate = line_candidates[i]
        ind = best_list[i]
        best_candidate.append(candidate[ind])
    return best_candidate


def replace_words(line, word_list, new_word_cand_list):
    new_line = ""
    current_index = 0
    for i in range(len(word_list)):
        old_word = word_list[i]
        new_word = new_word_cand_list[i][0]
        start_index = line.find(old_word, current_index)
        if start_index == -1:
            start_index = current_index
            curr_char = line[start_index]
            while curr_char == ' ':
                start_index += 1
                curr_char = line[start_index]
        old_index = current_index
        current_index = start_index + len(old_word)
        string_seq = line[old_index:current_index]
        string_seq = string_seq.replace(old_word, new_word)
        new_line += string_seq
        if current_index >= len(line):
            break
    return new_line


def get_best_candidate(prob_list):
    '''
    Get the candidates with highest probability
    :param prob_list: a list of cadidates to be selected
    :return: the index of the best element
    '''
    best_prob = float('-Inf')
    best_index = 0
    for i in range(len(prob_list)):
        if prob_list[i][1] > best_prob:
            best_prob = prob_list[i][1]
            best_index = i
    return best_index


def get_unigram_prob(unigram_cand, unigram_dict, delta, total_takens):
    '''
    Calculate the probability of unigram, add delta smoothing strategy is applied
    :param unigram_cand: the unigram candidata to be calculated
    :param unigram_dict: the dictionary of all unigrams
    :param delta: value of delta to be used
    :param total_takens: the size of the training corpus
    :return: the probability of the unigram
    '''
    unigram = unigram_cand[0]
    cost = unigram_cand[1]
    voc_size = len(unigram_dict)
    cw = delta
    unigram_low = unigram.lower()
    if unigram_low in unigram_dict:
        cw += unigram_dict[unigram_low]
    total = delta * voc_size + total_takens
    return float(cw) / (total * cost)


def get_bigram_prob(bigram_cand, unigram_dict, bigram_dict, delta, total_tokens):
    '''
    Calculate the probability of bigram, add delta smoothing strategy is applied
    :param bigram_cand: the bigram candidate to be calculated
    :param unigram_dict: the dictionary of all unigrams
    :param bigram_dict: the dictionary of all bigrams
    :param delta: value of delta to be used
    :return: the probability of the bigram
    '''
    bigram = (bigram_cand[0][0], bigram_cand[1][0])
    bigram_cost = (bigram_cand[0][1], bigram_cand[1][1])
    first = bigram[0].lower()
    second = bigram[1].lower()
    voc_size = len(unigram_dict)
    cw = delta
    bigram_low = (first, second)
    has_bigram = False
    if bigram_low in bigram_dict:
        cw += bigram_dict[bigram_low]
        has_bigram = True
    elif second in unigram_dict:
        cw += unigram_dict[second]
    total = delta * voc_size
    if has_bigram and first in unigram_dict:
        total += unigram_dict[first]
    else:
        total += total_tokens
    return float(cw) / (total * bigram_cost[1])


def get_candidate(err_word, unigram_dic, topN, threshold):
    """
    Find N candidates of a error word with the lowest cost
    :param err_word: the word to be corrected
    :param unigram_dic: dictionary of unigrams
    :param topN: the number of candidate to be selected
    :return: a list of candidate and its cost
    """
    candidate = []
    err_word_low = err_word.lower()
    if topN > (len(unigram_dic) + 1):
        candidate = list(unigram_dic.keys())
        candidate.append(err_word)
        return candidate
    else:
        for word in unigram_dic.keys():
            cost = wf_levenshtein(err_word_low, word)
            if err_word[0].isupper():
                word = word.capitalize()
            if cost > threshold:
                continue
            cost += 1
            if len(candidate) < topN:
                candidate.append((word, cost))
            else:
                biggest_index, biggest_cost = find_biggest(candidate)
                if cost < biggest_cost:
                    candidate[biggest_index] = (word, cost)
    """
    final_cand = []
    final_cand.append(err_word)
    for item in candidate:
        token = item[0]
        final_cand.append(token)
    return final_cand
    """
    candidate = clean_candidate(candidate)
    candidate.append((err_word, 1))
    return candidate


def clean_candidate(candidate):
    ind, lowest_cost = find_lowest(candidate)
    new_candidate = []
    for item in candidate:
        if item[1] == lowest_cost:
            new_candidate.append(item)
    return new_candidate


def find_lowest(items):
    '''
    Find the item with lowest cost
    :param items: a list of items to be search
    :return: the index of the selected item
    '''
    index = 0
    cost = float("Inf")
    for i in range(0, len(items)):
        curr_cost = items[i][1]
        if curr_cost < cost:
            cost = curr_cost
            index = i
    return index, cost


def find_biggest(items):
    '''
    Find the item with highest cost
    :param items: a list of items to be search
    :return: the index of the selected item
    '''
    index = 0
    cost = -1
    for i in range(0, len(items)):
        curr_cost = items[i][1]
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
    new_line = []
    for line in train_file.readlines():
        if re.match("\\s+", line):
            continue
        new_line.append(modify_line(unigram, bigram, line, total_tokens, TOKENIZER, 10, 1))
