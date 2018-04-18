"""
    A simple reader which reads plain noisy text.

    Author: Haobo Gu
    Date created: 04/15/2018
    Python version: 3.6.2
"""


class Reader:
    data_path = '../data/'

    def __init__(self, filename):
        self.filename = self.data_path + filename

    def read_all(self):
        """
        Read all contents in file
        :return: list of strings, one string per line
        """

        import codecs
        with codecs.open(self.filename, "r", encoding='utf-8', errors='ignore') as input_file:
            contents = input_file.read()
            contents = contents.split('\r\n')
        return contents


# Test script

# r = Reader('newberry-mary-b-some-further-accounts-of-the-nile-1912-1913.txt')
# content = r.read_all()
# print(content)
# for item in content:
#     print(item)
