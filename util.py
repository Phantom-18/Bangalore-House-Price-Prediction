import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    print("loading saved artifacts...")

    with open("Server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # sqft, bath, bhk excluded

    with open("Server/artifacts/bangalore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)

    print("artifacts loaded successfully")


# 🔥 THIS IS THE KEY FIX
# Load artifacts immediately when util is imported by Flask
load_saved_artifacts()


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)
