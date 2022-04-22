# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd
import matplotlib.pyplot as plt



def clean_data(model_output = None):
    if model_output is None:
        model_output = pd.read_csv("freq_baseline.tsv", sep="\t")
    model_output.columns=["word", "label", "prediction"]
    #print(model_output)
    model_output = model_output.dropna()
    #print(model_output.head())
    #model_output.rename(columns={"word", "label", "prediction"}, inplace=True)
    #print(model_output)
    #model_output.to_csv("model_out.csv")
    return model_output


if __name__ == '__main__':
    weighted_avg = []
    # for i in range(5):
    #Per class, per model: Precision, Recall, F1
    model_output = pd.read_csv("majority_baseline.tsv", sep="\t")
    model_output = clean_data(model_output)

    C = model_output[model_output["label"] == "C"]
    #print(C)
   # C.to_csv("c.tsv")
    N = model_output[model_output["label"] == "N"]
    #N.to_csv("n.tsv")
    #print(N)

    #Precision = True Positives / (True Positives + False Positives)

    trueposC = C[C["label"] == C["prediction"]].shape[0]
    falseposC =  N[N["prediction"] == "C"].shape[0]

    trueposN = N[N["label"] == N["prediction"]].shape[0]
    falseposN = C[C["prediction"] == "N"].shape[0]
    #print("WHAT", trueposC)
    #print("Falsepos", falseposC)
    if trueposC + falseposC == 0:
        precisionC = 0
    else:
        precisionC =  trueposC / (trueposC + falseposC)
    #print(precisionC)

    precisionN = trueposN / (trueposN + falseposN)
    #print(trueposN, falseposN, precisionN)


    #Recall = True Positives / (True Positives + False Negatives)
    falsenegC = C[C["prediction"] == "N"].shape[0]
    recallC = trueposC / (trueposC + falsenegC)
    #print(trueposC, falsenegC, recallC)

    falsenegN = N[N["prediction"] == "C" ].shape[0]
    recallN = trueposN / (trueposN + falsenegN)
    #print(trueposN, falsenegN, recallN)

     #F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if precisionC + recallC == 0:
        F1C = 0
    else:   F1C = 2 * (precisionC * recallC) / (precisionC + recallC)
    #
    F1N = 2 * (precisionN * recallN) / (precisionN + recallN)
    #
    #
    #print("Precision C = ", precisionC,  "  N = ", precisionN)
    #print("Recall C = ", recallC, " N = ", recallN)

    weighted = (F1C * C.shape[0] + F1N * N.shape[0]) / model_output.shape[0]
    print(precisionN, recallN, F1N, precisionC, recallC, F1C, weighted)
    # print(weighted_avg)
    # plt.plot(weighted_avg)
    # xtic = [0, 1, 2, 3, 4]
    # labels = ['50', '100', '150', '200', '300']
    # plt.title("Weighted F1-scores with varying embedding dimension")
    # plt.xticks(xtic, labels)
    # plt.ylabel("Weighted F1-score")
    # plt.savefig("14.png")
    # plt.show()
    #
    # ## TO - DO weighthed average

