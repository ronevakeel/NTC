"""
Provides evaluation methods for OCR correction result.
"""
import nltk
import src.wer as wer
import pyrouge


def evaluate(result_string, gold_string, eval_method='wer'):
    """
    Calculate overall word match rate between result string and gold string.
    WMR is an evaluation measurement with considering order of the word in the sentence
    :type result_string: str
    :type gold_string: str
    :type eval_method: str
    :rtype: float
    """
    if eval_method == 'wer':
        hyp_list = result_string.split()
        ref_list = gold_string.split()
        # print(word_list1, word_list2)
        eval_result = wer.calculate_wer(ref_list, hyp_list)
    # elif eval_method == 'rouge':
    #     # pyrouge.Rouge155.convert_summaries_to_rouge_format(system_input_dir, system_output_dir)
    #     return
    return eval_result

def evaluate_file(hyp_filename, ref_filename, eval_method='wer'):
    """
    Calculate overall word match rate between result string and gold string.
    WMR is an evaluation measurement with considering order of the word in the sentence
    :type result_string: str
    :type gold_string: str
    :type eval_method: str
    :rtype: float
    """
    if eval_method == 'wer':
        return 0
        # hyp_list = result_string.split()
        # ref_list = gold_string.split()
        # # print(word_list1, word_list2)
        # eval_result = wer.calculate_wer(ref_list, hyp_list)
    elif eval_method == 'rouge':
        r = pyrouge.Rouge155()
        r.system_dir = 'path/to/system_summaries'
        r.model_dir = 'path/to/model_summaries'
        r.system_filename_pattern = 'some_name.(\d+).txt'
        r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'

        eval_result = r.convert_and_evaluate(hyp_filename, ref_filename)
    return eval_result


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
    hyp = 'This machine can wreck a nice beach'
    ref = 'This great machine can recognize speech'
    print(evaluate(hyp, ref))
