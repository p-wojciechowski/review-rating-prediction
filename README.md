# Hotel review's rating prediction model

This project uses Keras LSTM model to predict rating given along with hotel review based on its text. [Trip Advisor Hotel Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) dataset was used to train the model. Model uses pre-train word vectors created with [GloVe](https://nlp.stanford.edu/projects/glove/) algorithm to create embedding layer.

Final model is deployed on https://reviews-rating-predictor.herokuapp.com/ as REST API with FastAPI.

Summary of EDA and model selection is [here](https://github.com/p-wojciechowski/review-rating-prediction/tree/main/notebooks).

### Usage

To get a prediction, you need to send a POST request to the `/predict` endpoint. Place json that contains `{"review_text":"<text of review>"}` in request's body. Example with curl:

```bash
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"review_text":"<text of review>"}' \
  https://reviews-rating-predictor.herokuapp.com/predict
```

> The app is hosted on a free plan of Heroku, therefore application is switched to sleep mode after 30 minutes of inactivity. You may need to wake app instance up, you can do this with a GET request sent to `/status` endpoint. After a few moments app should be operating.

For more specific API documentation, see https://reviews-rating-predictor.herokuapp.com/docs , where you can also send POST requests to get prediction result (in case you don't have a tool to send POST requests).

### Content

* `main.py` - main Python file with FastAPI app
* `model.py` - Python file with model loading and it's functions
* `/notebooks` - folder with Jupyter notebooks with EDA/preprocessing and model selection
* `/lstm_model` - folder with prediction model and preprocessing objects
* `Procfile` - file with command to run on Heroku
* `runtime.txt` - file to specify runtime on Heroku
* `requirements.txt` - requirements suited for deploying on Heroku (do not include libraries required to EDA and training)
* `requirements_training.txt`- requirements for EDA and model training

### Tools

* Keras/Tensorflow - building model
* Keras Tuner - searching hyperparameters
* FastAPI - API framework

