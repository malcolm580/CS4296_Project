import flask
from flask import render_template, request
import sys
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

from keras.models import load_model


class Recommand():
    def __init__(self, userID ):
        self.userID = userID
        # self.model = model

    def run(self):
        result = {}

        PATH = './ml-latest-small/'
        ratings = pd.read_csv(PATH + 'ratings.csv')
        model = load_model('movie_model.h5', compile=False)
        user_enc = LabelEncoder()
        ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
        n_users = ratings['user'].nunique()
        item_enc = LabelEncoder()
        ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)
        # print(ratings)

        movie_data = np.array(list(set(ratings['movie'])))
        # print(movie_data , len(movie_data))
        user = np.array([self.userID for i in range(len(movie_data))])
        print("Recommend top 5 movie to User ID#", self.userID)
        result['userID'] = self.userID

        predictions = model.predict([user, movie_data])
        predictions = np.array([a[0] for a in predictions])
        movie_ids = (-predictions).argsort()[:5]
        print("Movie ID", movie_ids)
        result['movie_ids'] =  movie_ids
        print("Predicted User rating", predictions[movie_ids])
        result['prediction'] = predictions[movie_ids]

        return result


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST', 'GET'])
def home():
    start_time = time.time()
    data = {'success': False}
    print('request')
    userID = request.args.get('userID')
    r = Recommand(userID)
    result = r.run()
    return render_template('index.html', variable=result , time= (time.time() - start_time) )

if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
        'please wait until server has fully started'))
    app.run( host='0.0.0.0', debug = False , threaded=False)
