from pandas import DataFrame
from sklearn.externals import joblib
import numpy as np

from utils import load_df


def create_submission(ids, predictions, filename='../out/submission_unit.csv'):
    submissions = np.concatenate((ids.reshape(len(ids), 1), predictions.reshape(len(predictions), 1)), axis=1)
    df = DataFrame(submissions)
    df.to_csv(filename, header=['id', 'click'], index=False)


print("*******************test starting***************************")

classifier = joblib.load('../out/model/classifier_unit.pkl')
# test_data_df = load_df('csv/test', training=False)
test_data_df = load_df('../datasets/test/test', training=False)
ids = test_data_df.values[0:, 0]

print(ids)

predictions = classifier.predict(test_data_df.values[0:, 1:])
create_submission(ids, predictions)

print("*******************test end***************************")


