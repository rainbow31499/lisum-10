import random
from datetime import datetime
import pandas as pd

# A sample data of 40000000 observations of soil temperatures taken at 538 locations around the world

random.seed(1030)

indices = list(range(40000000))

location = [random.randrange(1,539) for i in indices]
location_temperature = {i:random.uniform(-10,30) for i in range(1,539)}
temperature = [random.normalvariate(location_temperature[location[i]], 4) for i in indices]
depth = [random.uniform(15,35) for i in indices]
time_taken = [datetime.fromtimestamp(random.uniform(datetime(2022,6,1,0,0,0).timestamp(), \
    datetime(2022,7,1,0,0,0).timestamp())).isoformat() for i in indices]

data = pd.DataFrame({"location": location, \
                     "temperature": temperature, \
                     "depth": depth, \
                     "time_taken": time_taken})

data.to_csv("sample_data.csv", index=False)