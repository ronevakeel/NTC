import operator
import re
import os
import nltk

data_path = "../data/local/"
model_path = "../output/"
WHITE_SPACE = 0
TOKENIZER = 1

def readfile(file_name):
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
    uni_file = open(os.path.join(path, "unigram"), 'w')
    bi_file = open(os.path.join(path, "bigram"), 'w')

    for item, value in unigram.items():
        uni_file.write(item + " " + str(value) + "\n")
    for item, value in bigram.items():
        bi_file.write(item[0] + " " + item[1] + " " + str(value) + "\n")


def get_files(dir, file_list):
    files = os.listdir(dir)
    for file in files:
        if operator.eq(file, ".DS_Store"):
            continue
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            file_list.append(path)
        elif os.path.isdir(path):
            get_files(path, file_list)


def count_appearance(content, ugram_dict, bgram_dict, split_strategy=WHITE_SPACE):
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
        count_appearance(contentlist, unigramdict, bigramdict)
    writefile(unigramdict, bigramdict, output_path)


if __name__ == "__main__":
    ngrammodel(data_path)
