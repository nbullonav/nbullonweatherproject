import os
import pandas as pd
from datetime import datetime
from meteostat import daily, monthly, Point, stations

station_id = os.environ["STATION_ID"]

start = datetime(1980, 1, 1)
end = datetime(2024, 12, 31)

metadata_sta = stations.meta(station_id)
df_metadata = pd.DataFrame([metadata_sta])

data = monthly(station_id, start, end)
data = data.fetch()

data['id'] = str(station_id)
data.reset_index(inplace=True)

finaldata = pd.merge(data,df_metadata, how='left', on='id')
finaldata.drop(columns=['id','identifiers','timezone'], inplace=True)
finaldata.set_index('time', inplace=True)

whichfields = ['temp','tmin','tmax','prcp','pres','name','country','latitude','longitude','elevation']
# Define csv filename
csvfile = 'station' + station_id + '.csv'

df.to_csv(csvfile)
print(f"Guardado: {csvfile}")
