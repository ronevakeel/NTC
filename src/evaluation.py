"""
Provides evaluation methods for OCR correction result.

    Author: Haobo Gu
    Date created: 04/24/2018
    Python version: 3.6.2
"""
import nltk
import src.wer as wer
import re


def evaluate(result_string, gold_string, eval_method='wer'):
    """
    Evaluate system's performance using word error rate or rouge measurement.
    Use 'wer' and 'rouge' to specify which measurement you're going to use.
    :type result_string: str
    :type gold_string: str
    :type eval_method: str
    :rtype: float list
    """

    if eval_method == 'wer':
        # hyp_list = result_string.split()
        # ref_list = gold_string.split()
        re_list, gold_list = split_by_year(result_string, gold_string)

        total_errors = 0
        total_len = 0
        for i in range(len(re_list)):
            re_seq = re_list[i].split()
            gold_seq = gold_list[i].split()
            word_error_rate = wer.calculate_wer(re_seq, gold_seq)
            total_errors += len(gold_seq) * word_error_rate
            total_len += len(gold_seq)
        eval_result = total_errors / total_len

    return eval_result


# def evaluate_by_folder(result_folder, gold_folder, result_name_pattern, gold_name_pattern, eval_method='rouge'):
#     """
#     Evaluate system's performance using word error rate or rouge measurement.
#     All files in the folder will be read.
#     Use 'wer' and 'rouge' to specify which measurement you're going to use.
#     :type result_folder: str
#     :type gold_folder: str
#     :type eval_method: str
#     :rtype: dictionary
#     """
#     if eval_method == 'wer':
#         return 0
#         # hyp_list = result_string.split()
#         # ref_list = gold_string.split()
#         # # print(word_list1, word_list2)
#         # eval_result = wer.calculate_wer(ref_list, hyp_list)

#     return eval_result


def split_by_year(result_text, gold_text):
    """
    Split the result string and gold string by year, so that the time complicity of evaluation will significantly reduce.
    :type result_text: str
    :type gold_text: str
    :return: split list of result and the split list of gold standard
    """
    result_text_list = []
    gold_text_list = []
    result_list = re.split("(1912)", result_text)
    gold_list = re.split("(1912)", gold_text)
    i = 1
    while i < len(result_list):
        result_text_list.append(result_list[i-1] + result_list[i])
        gold_text_list.append(gold_list[i-1] + gold_list[i])
        i += 2
    re_text = result_list[len(result_list) - 1]
    go_text = gold_list[len(gold_list) - 1]
    result_list = re.split("(191[3;)])", re_text)
    gold_list = re.split("(1913)", go_text)
    if len(gold_list) > 35:
        gold_list[32] += gold_list[33] + gold_list[34]
        del gold_list[33]
        del gold_list[33]
    i = 1
    while i < len(result_list):
        result_text_list.append(result_list[i-1] + result_list[i])
        gold_text_list.append(gold_list[i-1] + gold_list[i])
        i += 2
    return result_text_list, gold_text_list

"""
** NOT COMPLETED PART **
"""
def word_match_rate(result_string, gold_string, scope=10):
    """
    Calculate overall word match rate between result string and gold string.
    WMR is an evaluation measurement with considering order of the word in the sentence

    :type result_string: str
    :type gold_string: str
    :type scope: int
    :rtype: float
    """
    re_word_list = nltk.word_tokenize(result_string)
    gold_word_list = nltk.word_tokenize(gold_string)
    gold_len = len(gold_word_list)
    cur = 0  # cursor in gold word list
    for index, word in enumerate(re_word_list):
        # Get range of
        start_index = cur - scope if cur - scope >= 0 else 0
        end_index = cur + scope if cur + scope < gold_len else gold_len


# Test script
if __name__ == '__main__':
    '''
    hyp = 'This machine can wreck a nice beach'
    # ref = 'This great machine can recognize speech'
    ref = 'This machine can wreck a good beach'
    print(evaluate(hyp, ref, 'wer'))
    '''
    raw_ocr_file = open("../data/OCR_text/newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt", 'r')
    gold_standard = open("../data/gold_standard/Edit_ Some Further Accounts of the Nile 1912-1913.txt", 'r')

    raw_content = raw_ocr_file.read()
    gold_content = gold_standard.read()

    raw_parts, gold_parts = split_by_year(raw_content, gold_content)
    for i in range(len(raw_parts)):
        rw = open("../data/OCR_text/NMB_" + str(i) + ".txt", 'w')
        gw = open("../data/gold_standard/ESF_" + str(i) + ".txt", 'w')
        rw.write(raw_parts[i])
        gw.write(gold_parts[i])
