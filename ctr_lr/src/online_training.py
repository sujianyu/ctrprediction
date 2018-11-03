import numpy
from sklearn import linear_model
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from train import save_model
from train import save_model_append_name
from utils import clean_df
import pandas as pd

# train_data_path = '../datasets/train/train'
train_data_path = '../datasets/train/train_small.csv'
# df = pd.read_csv('../datasets/train/train', header=0, chunksize=1)
df = pd.read_csv(train_data_path, header=0, chunksize=1)


global_X_test = numpy.ndarray((0,13))
global_y_test = numpy.ndarray((0,))
clf = linear_model.SGDClassifier()

# counter = 0
counter = 1
for chunk in df:
	chunk = clean_df(chunk)
	train_data = chunk.values
	X_train, X_test, y_train, y_test = train_test_split(train_data[0::, 1::], train_data[0::, 0],
                                                    test_size=0.3, random_state=0)

	global_X_test = numpy.concatenate((global_X_test, X_test))
	global_y_test = numpy.concatenate((global_y_test, y_test))
	clf.partial_fit(X_train, y_train, classes=[0, 1])
	counter += 1

	if counter % 100 == 0 :
		# print ('Counter', counter)
		# print ('Score', clf.score(global_X_test, global_y_test))

		print ('Counter:', counter,' Score:', clf.score(global_X_test, global_y_test))
		# print (' Score:', clf.score(global_X_test, global_y_test))

# save_model(clf,'classifier_trained_online.pkl')
save_model_append_name(clf,'classifier_trained_online.pkl')

print ("******************************************************************************")
print ("clf_score:",clf.score(global_X_test, global_y_test))



