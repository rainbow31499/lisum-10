import pandas as pd
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

toy_data = pd.read_csv("all_perth_310121.csv")

toy_data["DATE_SOLD"] = toy_data["DATE_SOLD"].map(lambda x: datetime.strptime(x, "%m-%Y\r").toordinal())
toy_data["GARAGE"] = toy_data["GARAGE"].fillna(0)
toy_data["BUILD_YEAR"] = toy_data["BUILD_YEAR"].fillna(2022)
toy_data["NEAREST_SCH_RANK"] = toy_data["NEAREST_SCH_RANK"].fillna(0)

X = toy_data[["BEDROOMS", \
              "BATHROOMS", \
              "GARAGE", \
              "LAND_AREA", \
              "FLOOR_AREA", \
              "BUILD_YEAR", \
              #"CBD_DIST", \
              #"NEAREST_STN_DIST", \
              #"DATE_SOLD", \
              #"LATITUDE", \
              #"LONGITUDE", \
              #"NEAREST_SCH_DIST", \
              #"NEAREST_SCH_RANK" \
             ]]
y = toy_data["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

pickle.dump(linreg, open("model.pickle", "wb"))