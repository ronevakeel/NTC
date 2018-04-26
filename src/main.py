import src.file_reader as reader
import src.evaluation as evaluation

if __name__ == "__main__":

    data = reader.get_pairs("", "")
    print(evaluation.evaluate(data[0][:10000], data[1][:10000]))

