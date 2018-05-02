"""
    Noisy text correction module
"""
import os
import re
import nltk

class NoisyTextCorrection:
    def __init__(self, rbm):
        self.rbm = rbm

    def apply_all_rules(self, text):
        # apply rules on noisy text

        ruleset_list = ['CorrectionRules.txt', 'FusingRules.txt',
                        'HyphenRules.txt', 'SyncopeRules.txt', 'AmbiguousPairs.txt', 'VariantSpellings.txt']
        for rule_name in ruleset_list:
            self._apply_rule(text, rule_name)
        return text

    def _apply_rule(self, text, rule_type='CorrectionRules.txt'):
        """
        Apply selected rule on text.
        :param text: str
        :param rule_type: str
        :return: str
        """

        if rule_type == 'CorrectionRules.txt':
            for key, value in self.rbm.correction_rules.items():
                text = text.replace(key, value)
        elif rule_type == 'AmbiguousPairs.txt':
            for key, value in self.rbm.ambiguous_pairs.items():
                text = text.replace(key, value)
        elif rule_type == 'FusingRules.txt':
            for key, value in self.rbm.fusing_rules.items():
                text = text.replace(key, value)
        elif rule_type == 'HyphenRules.txt':
            for key, value in self.rbm.hyphen_rules.items():
                text = text.replace(key, value)
        elif rule_type == 'SyncopeRules.txt':
            for key, value in self.rbm.syncope_rules.items():
                text = text.replace(key, value)
        elif rule_type == 'VariantSpellings.txt':
            for key, value in self.rbm.variant_spelling.items():
                text = text.replace(key, value)

        return text

    def apply_char_rule(self, text):
        # apply rules on noisy text
        word_seq = nltk.word_tokenize(text)
        for word in word_seq:
            word = re.sub('^\W+$', '', word)
            corrected_word = word
            continue_loop = 1
            if word not in self.rbm.vocabulary:
                # Only process wrong word
                for key, sub_list in self.rbm.char_rules.items():
                    # Check every rule
                    if key in word:
                        for candidate in sub_list:
                            if word.replace(key, candidate) in self.rbm.vocabulary:
                                corrected_word = word.replace(key, candidate)
                                continue_loop = 0
                                break
                    if continue_loop == 0:
                        # word is changed
                        text = text.replace(word, corrected_word)
                        # print(word, corrected_word)
                        if word == '1912' or word == '1913' or word == '191;' or word == '191)':
                            print(word, corrected_word)
                        break

        return text

    def process(self, text):
        # main method to process noisy text

        # Apply rules first
        text = self.apply_char_rule(text)

        # Statistical approach
        return text


class RuleBasedModel:
    def __init__(self, ruleset_folder):
        """
        :param ruleset_folder: the folder name
        """
        self.ambiguous_pairs = {}
        self.correction_rules = {}
        self.fusing_rules = {}
        self.hyphen_rules = {}
        self.syncope_rules = {}
        self.variant_spelling = {}
        self.char_rules = {}
        self.vocabulary = []
        self.place_names = []
        self.personal_names = []
        self.dis_bigrams = []

        if ruleset_folder in os.listdir('.'):
            self.path = './' + ruleset_folder
        elif ruleset_folder in os.listdir('..'):
            self.path = '../' + ruleset_folder
        else:
            print('Wrong rule set folder name!')
            exit(0)
        # self.read_all_rules_and_vocab()
        self.read_char_rule_and_vocab()

    def read_char_rule_and_vocab(self):
        """
        Read char rules and vocabularies in to rule-based model.
        """
        ruleset_list = ['CharRules.txt', 'vocabulary.txt', 'PlaceNames.txt', 'PersonalNames.txt', 'unigram']
        for filename in ruleset_list:
            file_path = os.path.join(self.path, filename)
            ruleset = {}
            voc = []
            with open(file_path, 'r', encoding='utf-8') as r_f:
                content = r_f.read()
                rule_list = content.split('\n')
                # If not vocabulary
                if filename not in ['vocabulary.txt', 'PlaceNames.txt', 'PersonalNames.txt',
                                    'DisambigTwograms.txt', 'unigram']:
                    for rule in rule_list:
                        elements = rule.split(';')
                        if int(elements[2]) > 10:
                            if elements[1] in ruleset:
                                ruleset[elements[1]] = ruleset[elements[1]] + [elements[0]]
                            else:
                                ruleset[elements[1]] = [elements[0]]  # Store rule using dictionary
                    self.char_rules = ruleset
                else:
                    # Read vocabulary
                    for word in rule_list:
                        if filename == 'DisambigTwograms.txt':
                            voc.append(word.strip('\n').split('\t')[0])
                        else:
                            voc.append(word.strip('\n').split(' ')[0])
                    # if filename == 'vocabulary.txt':
                    if filename == 'unigram':
                        self.vocabulary = set(voc)
                    elif filename == 'PlaceNames.txt':
                        self.place_names = set(voc)
                    elif filename == 'PersonalNames.txt':
                        self.personal_names = set(voc)
                    elif filename == 'DisambigTwograms.txt':
                        self.dis_bigrams = set(voc)

    def read_all_rules_and_vocab(self):
        """
        Read all rules and vocabularies in to rule-based model.
        """
        ruleset_list = ['AmbiguousPairs.txt', 'CorrectionRules.txt', 'DisambigTwograms.txt', 'FusingRules.txt',
                        'HyphenRules.txt', 'SyncopeRules.txt', 'VariantSpellings.txt', 'vocabulary.txt',
                        'PlaceNames.txt', 'PersonalNames.txt']
        for filename in ruleset_list:
            file_path = os.path.join(self.path, filename)
            ruleset = {}
            voc = []
            with open(file_path, 'r', encoding='utf-8') as r_f:
                content = r_f.read()
                rule_list = content.split('\n')
                # If not vocabulary
                if filename not in ['vocabulary.txt', 'PlaceNames.txt', 'PersonalNames.txt', 'DisambigTwograms.txt']:
                    for rule in rule_list:
                        elements = rule.split('\t')
                        ruleset[elements[0]] = elements[1]  # Store rule using dictionary
                    if filename == 'AmbiguousPairs.txt':
                        self.ambiguous_pairs = ruleset
                    elif filename == 'CorrectionRules.txt':
                        self.correction_rules = ruleset
                    elif filename == 'FusingRules.txt':
                        self.fusing_rules = ruleset
                    elif filename == 'HyphenRules.txt':
                        self.hyphen_rules = ruleset
                    elif filename == 'SyncopeRules.txt':
                        self.syncope_rules = ruleset
                    elif filename == 'VariantSpellings.txt':
                        self.variant_spelling = ruleset
                else:
                    # Read vocabulary
                    for word in rule_list:

                        if filename == 'DisambigTwograms.txt':
                            voc.append(word.strip('\n').split('\t')[0])
                        else:
                            voc.append(word.strip('\n'))
                    if filename == 'vocabulary.txt':
                        self.vocabulary = voc
                    elif filename == 'PlaceNames.txt':
                        self.place_names = voc
                    elif filename == 'PersonalNames.txt':
                        self.personal_names = voc
                    elif filename == 'DisambigTwograms.txt':
                        self.dis_bigrams = voc


# Test script
if __name__ == '__main__':
    rbm = RuleBasedModel('ruleset')
    rbm.read_char_rule_and_vocab()
    for key in rbm.vocabulary:
        print(key)

