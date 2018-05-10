import src.file_io as reader
import src.evaluation as evaluation
import src.ntc as ntc
import src.ngram as ng
import re
import os
import nltk

if __name__ == "__main__":

    if 'output' in os.listdir('.'):
        data_path = './data/'
        output_path = "./output/"
        n_gram_data_path = "./data/local/"
    else:
        data_path = '../data/'
        output_path = "../output/"
        n_gram_data_path = "../data/local/"
    raw_text_file = "OCR_text/NMB_2.txt"
    gold_text_file = "gold_standard/ESF_2.txt"

    # Read data
    data = reader.read_file(data_path + raw_text_file)
    data = reader.clean_empty_line(data)

    # Rule-based system
    rbm = ntc.RuleBasedModel('ruleset')
    model = ntc.NoisyTextCorrection(rbm)
    result = []
    for line in data:
        result.append(model.process(line))

    # Statistical system
    unigram, bigram, total_tokens = ng.ngrammodel(n_gram_data_path, output_path)
    new_result = []
    for line in result:
        if not re.match("\\s+", line):
            new_line = ng.modify_line(unigram, bigram, line, total_tokens, ng.TOKENIZER, 50, 0.1, 1.7)
            new_result.append(new_line)
            print(new_line)

    # Write result
    reader.write_file(result, output_path+'corrected_text1')
    reader.write_file(new_result, output_path+'corrected_text2')

    # Evaluation
    result = reader.lines2string(result)
    new_result = reader.lines2string(new_result)
    data = reader.lines2string(data)
    gold = reader.read_file(data_path+gold_text_file)
    gold = reader.clean_empty_line(gold)
    gold = reader.lines2string(gold)
    #
    # print('finished')
    # re_list, gold_list = evaluation.split_by_year(result, gold)
    # # result, gold = reader.get_pairs(data_path+raw_text_file, data_path+gold_text_file)
    # print(re_list.__len__(), gold_list.__len__())

    print('WER in raw text:', evaluation.evaluate(data, gold))
    print('WER after rule-based system:', evaluation.evaluate(result, gold))
    print('WER after rule-based and statistical system', evaluation.evaluate(new_result, gold))

    gold, raw = reader.get_pairs(data_path + gold_text_file, data_path + raw_text_file)
    gold, corr = reader.get_pairs(data_path + gold_text_file, output_path + "corrected_text2")
    print("raw: " + str(evaluation.evaluate(gold, raw)))
    print("corrected: " + str(evaluation.evaluate(gold, corr)))
