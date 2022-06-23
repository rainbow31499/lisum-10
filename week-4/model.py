import pandas as pd
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

toy_data = pd.read_csv("all_perth_310121.csv")
# Source: https://www.kaggle.com/datasets/syuzai/perth-house-prices

# print(toy_data.info()) outputs:
# RangeIndex: 33656 entries, 0 to 33655
# Data columns (total 19 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   ADDRESS           33656 non-null  object
#  1   SUBURB            33656 non-null  object
#  2   PRICE             33656 non-null  int64
#  3   BEDROOMS          33656 non-null  int64
#  4   BATHROOMS         33656 non-null  int64
#  5   GARAGE            31178 non-null  float64 (null assumed 0)
#  6   LAND_AREA         33656 non-null  int64
#  7   FLOOR_AREA        33656 non-null  int64
#  8   BUILD_YEAR        30501 non-null  float64 (null assumed 2022)
#  9   CBD_DIST          33656 non-null  int64
#  10  NEAREST_STN       33656 non-null  object
#  11  NEAREST_STN_DIST  33656 non-null  int64
#  12  DATE_SOLD         33656 non-null  int64
#  13  POSTCODE          33656 non-null  int64
#  14  LATITUDE          33656 non-null  float64
#  15  LONGITUDE         33656 non-null  float64
#  16  NEAREST_SCH       33656 non-null  object
#  17  NEAREST_SCH_DIST  33656 non-null  float64
#  18  NEAREST_SCH_RANK  22704 non-null  float64 (null assumed 0)

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