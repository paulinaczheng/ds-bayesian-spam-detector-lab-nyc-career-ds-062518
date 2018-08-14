import pytest
from ipynb.fs.full.index import df, column_names, target, clean_df, X_train, X_test, y_train, y_test, accuracy, f1, confusion_matrix, training_preds, training_cm, testing_cm

def test_p1_training_cm():
    assert len(df) == 58
    assert str(type(df)) == 'pandas.core.frame.DataFrame'

def test_p2_training_cm():
    assert len(column_names) == 58
    assert 'is_spam' in column_names
    assert 'word_freq_make' in column_names

def test_p3_training_cm():
    assert target == df['is_spam']

def test_p4_training_cm():
    assert len(X_train) == len(y_train)
    assert len(X_train) == 3450
    assert len(X_test) == len(y_test)
    assert len(X_test) == 1151

def test_p5_training_cm():
    assert "{:.4}".format(accuracy) == "82.1"

def test_p6_training_cm():
    assert "{:.4}".format(f1) == "80.49"

def test_p7_training_cm():
    test_cm_expected = {"TP": 1, "TN": 2, "FP": 3, "FN": 4}
    test_predictions = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    test_labels = [1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    test_cm_actual = confusion_matrix(test_predictions, test_labels)
    assert test_cm_expected == test_cm_actual

def test_p8_training_cm():
    assert training_preds[:5] == [1, 0, 0, 1, 1]

def test_p9_training_cm():
    assert training_cm["TP"] == 1310
    assert training_cm["TN"] == 1524
    assert training_cm["FP"] == 567
    assert training_cm["FN"] == 49

def test_p10_training_cm():
    assert testing_cm["TP"] == 425
    assert testing_cm["TN"] == 520
    assert testing_cm["FP"] == 177
    assert testing_cm["FN"] == 49
