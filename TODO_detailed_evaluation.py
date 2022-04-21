# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd



def clean_data():
    model_output = pd.read_csv("majority_baseline.tsv", sep="\t", encoding='latin-1') #"experiments/base_model/model_output.tsv"
    model_output.columns=["word", "label", "prediction"]
    #print(model_output)
    model_output = model_output.dropna()
    print(model_output.head())
    #model_output.rename(columns={"word", "label", "prediction"}, inplace=True)
    #print(model_output)
    model_output.to_csv("model_out_maj.csv")



if __name__ == '__main__':

    #Per class, per model: Precision, Recall, F1
    # clean_data()
    model_output = pd.read_csv("model_out_maj.csv")

    C = model_output[model_output["label"] == "C"]
    print(C)
   # C.to_csv("c.tsv")
    N = model_output[model_output["label"] == "N"]
    #N.to_csv("n.tsv")
    print(N)

    #Precision = True Positives / (True Positives + False Positives)

    trueposC = C[C["label"] == C["prediction"]].shape[0]
    falseposC =  N[N["prediction"] == "C"].shape[0]

    trueposN = N[N["label"] == N["prediction"]].shape[0]
    falseposN = C[C["prediction"] == "N"].shape[0]
    print("TrueposC", trueposC)
    print("FalseposC", falseposC)

    if (trueposC + falseposC) == 0:
        precisionC = 0
    else:
        precisionC = trueposC / (trueposC + falseposC)
    print("precisionC", precisionC)

    precisionN = trueposN / (trueposN + falseposN)
    print("N truepos, falsepos, precision", trueposN, falseposN, precisionN)


    #Recall = True Positives / (True Positives + False Negatives)
    falsenegC = C[C["prediction"] == "N"].shape[0]
    recallC = trueposC / (trueposC + falsenegC)
    print("C truepos, falseneg, recall", trueposC, falsenegC, recallC)

    falsenegN = N[N["prediction"] == "C" ].shape[0]
    recallN = trueposN / (trueposN + falsenegN)
    print("N truepos, falseneg, recall", trueposN, falsenegN, recallN)

     #F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if precisionC == 0 or recallC == 0:
        F1C = 0
    elif precisionC != 0 or recallC != 0:
        F1C = 2 * (precisionC * recallC) / (precisionC + recallC)

    if precisionN == 0 or recallN == 0:
        F1N = 0
    elif precisionN != 0 or recallN != 0:
        F1N = 2 * (precisionN * recallN) / (precisionN + recallN)

    print("Precision C = ", precisionC,  "  N = ", precisionN)
    print("Recall C = ", recallC, " N = ", recallN)
    print("F1C", F1C, "F1N", F1N)

    ## TO - DO weighthed average

