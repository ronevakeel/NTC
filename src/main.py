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
        n_gram_data_path = "./data/corpus/"
    else:
        data_path = '../data/'
        output_path = "../output/"
        n_gram_data_path = "../data/corpus/"
    raw_text_file = "OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913/NMB_2.txt"
    gold_text_file = "gold_standard/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913/ESF_2.txt"

    all_raw_text_files, all_gold_standard_files = reader.get_all_evaluation_files()
    possible_name_entity_dict = ng.get_possible_NE_list(all_raw_text_files)
    # Read data
    data = reader.read_file(data_path + raw_text_file)
    data = reader.clean_empty_line(data)

    # Rule-based system
    print('Rule-based ...')
    rbm = ntc.RuleBasedModel('ruleset')
    model = ntc.NoisyTextCorrection(rbm)
    result = []
    for line in data:
        result.append(model.process(line))

    # Statistical system
    print("Statistical ...")
    # unigram, bigram, total_tokens = ng.ngrammodel(n_gram_data_path, output_path)
    statistic_model = ng.read_ngram_model(output_path, ng.TOKENIZER, 50, 0.1, 1.7, possible_name_entity_dict)
    new_result = []
    for line in result:
        if not re.match("\\s+", line):
            new_line = ng.modify_line(statistic_model, line)
            new_result.append(new_line)

    # Write result
    print('Writing result ...')
    reader.write_file(result, output_path+'corrected_text1')
    reader.write_file(new_result, output_path+'corrected_text2_judged2')

    # Evaluation
    print('Evaluation ...')
    result = reader.lines2string(result)
    new_result = reader.lines2string(new_result)
    data = reader.lines2string(data)
    gold = reader.read_file(data_path+gold_text_file)
    gold = reader.clean_empty_line(gold)
    gold = reader.lines2string(gold)

    print('WER in raw text:', evaluation.evaluate(data, gold))
    print('WER after rule-based system:', evaluation.evaluate(result, gold))
    print('WER after rule-based and statistical system', evaluation.evaluate(new_result, gold))


