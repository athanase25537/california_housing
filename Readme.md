# California Housing Price Prediction API

A RESTful API built with FastAPI that predicts California housing prices based on demographic and geographic input features, including a categorical variable ocean_proximity.

## Requirements

See requirements.txt

## Setup

1. clone projet:

    git clone https://github.com/athanase25537/california_housing.git

2. go to the main directory:

        cd california_housing

3. install dependencies:

    pip install -r requirements.txt

4. Make sure you've download the model "california_housing.pkl"

## Run the API:

1. Bash:

    fastapi dev api.py

2. See swagger for the api documentation (copy and paste to your browser):

    http://127.0.0.1:8083/docs


## Endpoints
1. GET /

Returns a welcome message.

Response:

    {

        "message": "Welcome to California Housing API"

    }

2. POST /predict

Predicts the housing price based on the provided input data.

Request Body (JSON):

    {

        "longitude": -122.23,
        
        "latitude": 37.88,
        
        "housing_median_age": 41.0,
        
        "total_rooms": 880.0,
        
        "total_bedrooms": 129.0,
        
        "population": 322.0,
        
        "households": 126.0,
        
        "median_income": 8.3252,
        
        "ocean_proximity": "INLAND"
    
    }

Valid values for ocean_proximity:

    "NEAR BAY"

    "LESS THAN ONE HOUR TO OCEAN"

    "INLAND"

    "NEAR OCEAN"

    "ISLAND"

Sample Response:

    {

        "status": "success",
        
        "prediction": 231400.55,
        
        "score": "81.69%"

    }

## Model

The file california_housing.pkl contains a machine learning model trained using scikit-learn RandomForestRegressor and should be placed in the same directory as main.py.

1. Data Validation

    All numeric fields must be non-negative.

    A 400 Bad Request error is returned if an invalid ocean_proximity value is provided.

2. Project Structure

.
├── api.py
├── california_housing.pkl
├── train
└── README.md
