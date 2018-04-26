import src.file_io as reader
import src.evaluation as evaluation

if __name__ == "__main__":

    data = reader.get_pairs()
    print(evaluation.evaluate(data[0], data[1]))

