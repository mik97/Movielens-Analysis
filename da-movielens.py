import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from imblearn.over_sampling import RandomOverSampler


def main():

    movies = pd.read_csv(f'datasets/movies.csv')
    print(movies)

    ratings = pd.read_csv(f'datasets/ratings.csv')
    print(ratings)

    averageRateings = ratings.groupby(by='movieId').mean()

    toAdd = []
    for id in movies['movieId']:
        if id in averageRateings.index:
            toAdd.append(np.round(averageRateings.loc[id, 'rating']))
        else:
            toAdd.append(None)

    movies['averageRate'] = toAdd

    newDf = movies.dropna(axis=0)
    newDf['averageRate'] = newDf['averageRate'].astype(int)

    genres = newDf['genres'].str.get_dummies(sep='|')

    # oversample = RandomOverSampler(sampling_strategy='minority')
    # X_over, y_over = oversample.fit_resample(genres, newDf['averageRate'])
    # print(X_over)
    # print(y_over)

    dfFinal = pd.concat(
        [newDf[['movieId', 'title', 'averageRate']], genres], axis=1)
    print(dfFinal[dfFinal['averageRate'] == 0])

    # X_train, X_test, y_train, y_test = train_test_split(dfFinal[list(
    #     genres.axes[1])], dfFinal['averageRate'], test_size=0.2, random_state=42)
    # knn = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [1,3,10,15,20,25,50,100,150,200,250,300,350,400]})
    # knn.fit(X_train, y_train)
    # print("Test set predictions: {}".format(knn.predict(X_test)))
    # print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))
    # print(knn.best_params_)

    # svmClf = svm.SVC(kernel='linear', C=1, class_weight='balanced')
    # svmClf.fit(X_train, y_train)
    # print("Test set accuracy: {:.2f}".format(svmClf.score(X_test, y_test)))


main()
