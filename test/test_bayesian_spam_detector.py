import pytest
# import ipython
from ipynb.fs.full.index import (df, column_names, target, clean_df, X_train,
X_test, y_train, y_test, accuracy, f1, confusion_matrix, training_preds,
training_cm, testing_cm)

def test_p9_training_cm():
    assert training_cm["TP"] == 1310
    assert training_cm["TN"] == 1524
    assert training_cm["FP"] == 567
    assert training_cm["FN"] == 49
