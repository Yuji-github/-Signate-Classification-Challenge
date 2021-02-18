import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == '__main__':
    dataset = pd.read_csv('train.tsv', sep='\t')
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute.fit(X)
    X = impute.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    test = dataset = pd.read_csv('test.tsv', sep='\t')
    # print(test)
    num = test.iloc[:, 0].values
    test = test.values
    test = sc.transform(test[:, 1:])
    y_pred = classifier.predict(test)

    y_p = y_pred.reshape(len(y_pred), 1)
    num = num.reshape(len(num), 1)
    p = np.concatenate((num, y_p), axis=1)
    p = p.astype(int)

    cm = confusion_matrix(y_test, classifier.predict(X_test))
    print(cm)
    print(p)

    print(accuracy_score(y_true=y_test, y_pred=classifier.predict(X_test)))
    pd.DataFrame(p).to_csv("result.csv", index=False)