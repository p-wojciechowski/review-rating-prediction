# Review's rating prediction model

This project uses Keras CNN model to predict rating given along with review based on its text. [Trip Advisor Hotel Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) dataset was used to train the model. Model uses pre-train word vectors created with [GloVe](https://nlp.stanford.edu/projects/glove/) algorithm to create embedding layer. Final model is deployed on https://reviews-rating-predictor.herokuapp.com/ as REST API with FastAPI.

### Usage

To get a prediction, you need to send POST request to the `/predict` endpoint. Place json that contain `{"review_text":"<text of review>"}` in request's body. Example with curl:

```bash
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"review_text":"<text of review>"}' \
  https://reviews-rating-predictor.herokuapp.com/predict
```

> App is hosted on free plan of Heroku, therefore application is switched to sleep mode after 30 minutes of inactivity. It is possible that you will need to wake app instance up, you can do this with GET request sent to `/status` endpoint. After few moments app should be operating.

For more specific API documentation, see https://reviews-rating-predictor.herokuapp.com/docs , where you can also send POST requests to get prediction result (in case you don't have tool to send POST requests).

### Content

* `main.py` - main Python file with FastAPI app
* `model.py` - Python file with model loading and it's functions

* `notebooks` - folder with Jupyter notebooks with EDA/preprocessing and model selection
* `cnn_model` - folder with prediction model and preprocessing objects
* `Procfile` - file with command to run on Heroku
* `runtime.txt` - file to specify runtime on Heroku
* `requirements.txt` - requirements suited for deploying on Heroku (do not include libraries required to EDA and training)

### Stack

* `tensorflow==2.4.1` for EDA and training, `tensorflow-cpu==2.4.1` for deploy on Heroku
* `fastapi==0.63.0`
