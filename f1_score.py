import os
from sklearn.metrics import f1_score
y_true = []
y_pred = []

pred = open("/home/siva/siva-optimus/seq2seq-dec17-final/seq2seq-bh/MODEL_DIR/pred/predictions-700K.txt", "r")

with open("/home/siva/siva-optimus/seq2seq-dec17-final/seq2seq-bh/data/test_labels_sentences","r") as f:
    for line in f:
        line = line.split()
        labels = pred.readline().split()
        for i in range(len(line)):
            if i < len(labels):
                y_pred+=[labels[i]]
            else:
                y_pred+=["o"]
            y_true+=[line[i]]
score = f1_score(y_true, y_pred, average=None)
for s in score:
    print(s)
print("\n")
print(f1_score(y_true,y_pred, average='macro'))
               