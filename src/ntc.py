"""
    Noisy text correction module
"""
import os

class NoisyTextCorrection:
    def __init__(self, rule_set):
        self.rule_set = rule_set

    def apply_rules(self, text):
        # apply rules on noisy text
        return text

    def process(self, text):
        # main method to process noisy text
        return text


class RuleBasedModel:
    def __init__(self, ruleset_folder):
        """
        :param ruleset_folder: the foldername
        """
        self.ambiguous_pairs = {}
        self.correction_rules = {}
        self.fusing_rules = {}
        self.hyphen_rules = {}
        self.syncope_rules = {}
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

        ruleset_list = ['AmbiguousPairs.txt', 'CorrectionRules.txt', 'DisambigTwograms.txt', 'FusingRules.txt',
                        'HyphenRules.txt', 'SyncopeRules.txt', 'VariantSpellings.txt', 'vocabulary.txt',
                        'PlaceNames.txt', 'PersonalNames.txt']
        # for filename in ruleset_list:
        if True:
            filename = 'PersonalNames.txt'
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
                else:
                    # Read vocabulary
                    for word in rule_list:
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
    print(rbm.personal_names)

