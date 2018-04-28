"""
    Noisy text correction module
"""
import os
import re


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

    def _apply_rule(self, text, rule_type='CorrectionRules'):
        """
        Apply selected rule on text.
        :param text: str
        :param rule_type: str
        :return: str
        """

        if rule_type == 'CorrectionRules':
            for key, value in self.rbm.correction_rules:
                text = text.replace(key, value)
        elif rule_type == 'AmbiguousPairs.txt':
            for key, value in self.rbm.ambiguous_pairs:
                text = text.replace(key, value)
        elif rule_type == 'FusingRules.txt':
            for key, value in self.rbm.fusing_rules:
                text = text.replace(key, value)
        elif rule_type == 'HyphenRules.txt':
            for key, value in self.rbm.hyphen_rules:
                text = text.replace(key, value)
        elif rule_type == 'SyncopeRules.txt':
            for key, value in self.rbm.syncope_rules:
                text = text.replace(key, value)
        elif rule_type == 'VariantSpellings.txt':
            for key, value in self.rbm.variant_spelling:
                text = text.replace(key, value)

        return text

    def process(self, text):
        # main method to process noisy text

        # Apply rules first
        text = self.apply_all_rules(text)

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
        self.read_all_rules_and_vocab()

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
    rbm.read_all_rules_and_vocab()
    print(rbm.correction_rules)

