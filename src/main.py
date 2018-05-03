import src.file_io as reader
import src.evaluation as evaluation
import src.ntc as ntc

if __name__ == "__main__":
    data_path = '../data/'
    output_path = "../output/"
    raw_text_file = "OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt"
    cleaned_text_file = "cleaned_newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt"
    gold_text_file = "gold_standard/Edit_ Some Further Accounts of the Nile 1912-1913.txt"

    data = reader.read_file(data_path + raw_text_file)
    data = reader.clean_empty_line(data)

    rbm = ntc.RuleBasedModel('ruleset')
    model = ntc.NoisyTextCorrection(rbm)
    result = []
    for line in data:
        result.append(model.process(line))

    reader.write_file(result, output_path+'corrected_text1')
    result = reader.lines2string(result)
    data = reader.lines2string(data)
    gold = reader.read_file(data_path+gold_text_file)
    gold = reader.clean_empty_line(gold)
    gold = reader.lines2string(gold)

    print('finished')
    # re_list, gold_list = evaluation.split_by_year(result, gold)
    # # result, gold = reader.get_pairs(data_path+raw_text_file, data_path+gold_text_file)
    # print(re_list.__len__(), gold_list.__len__())
    # print(evaluation.evaluate(result, gold))

