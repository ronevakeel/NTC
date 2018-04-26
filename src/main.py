import src.file_io as reader
import src.evaluation as evaluation
import src.ntc as ntc


if __name__ == "__main__":

    data = reader.get_pairs()
    model = ntc.NoisyTextCorrection('')
    result = model.process(data[0])
    # print(result)
    print(evaluation.evaluate(result, data[1]))

