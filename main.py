import sys
import pandas as pd
from meteostat import Daily
from datetime import datetime

station_id = sys.argv[1]

start = datetime(1980, 1, 1)
end = datetime(2024, 12, 31)

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
