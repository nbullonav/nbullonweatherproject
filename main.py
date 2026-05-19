import os
import pandas as pd
from datetime import datetime
from meteostat import daily, monthly, Point, stations

station_id = os.environ["STATION_ID"]

start = datetime(1980, 1, 1)
end = datetime(2026, 4, 30)

metadata_sta = stations.meta(station_id)
df_metadata = pd.DataFrame([metadata_sta])

data = monthly(station_id, start, end)
data = data.fetch()

data['id'] = str(station_id)
data.reset_index(inplace=True)

finaldata = pd.merge(data,df_metadata, how='left', on='id')
finaldata.drop(columns=['id','elevation','tsun','identifiers','timezone'], inplace=True)
finaldata.set_index('time', inplace=True)

## 1.2. Captura de datos - Dataset 2 (ONI data de NOAA.gov)
df_sst = pd.read_csv("https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii", sep='\s+',
                     names=["YR", "MON", "NINO12","NINO12ANOM", "NINO3","NINO3ANOM", "NINO4","NINO4ANOM", "NINO34","NINO34ANOM"], header=0)
df_sst_final = df_sst[['YR','MON','NINO12',"NINO12ANOM","NINO34","NINO34ANOM"]]
df_sst_final['time'] = df_sst_final['YR'].astype(str) + '-' + df_sst_final['MON'].astype(str)
df_sst_final = df_sst_final[df_sst_final['time']>= "1980-01-01"]
df_sst_final['time'] = pd.to_datetime(df_sst_final['time'], format='%Y-%m')
df_sst_final.set_index('time', inplace=True)
df_sst_final.drop(columns=['YR','MON'], inplace=True)

# 1.3 Captura de datos - ICEN
df_icen = pd.read_csv("http://met.igp.gob.pe/datos/ICEN.txt", sep=r"\s+", skiprows=5 ,header=None,names=["YR","MON","ICEN"])
df_icen['time'] = df_icen['YR'].astype(str) + '-' + df_icen['MON'].astype(str)
df_icen = df_icen[df_icen['time']>= "1980-01-01"]
df_icen['time'] = pd.to_datetime(df_icen['time'], format='%Y-%m')
df_icen.set_index('time', inplace=True)
df_icen.drop(columns=['YR','MON'], inplace=True)

df_final = (
    finaldata
    .merge(df_sst_final, on="time", how="outer")
    .merge(df_icen, on="time", how="outer")
    )

#### EDA ####

### PRCP
max_prcp = df_final['prcp'].quantile(0.999)
max_prcp
min_prcp = df_final['prcp'].quantile(0.001)
min_prcp
df2 = df_final[df_final['prcp'] > min_prcp]
prcp_avg = df2['prcp'].mean().round(2)
df_final.loc[df_final['prcp'].isna(), 'prcp'] = prcp_avg
### TEMP
max_Temp = df_final['temp'].quantile(0.999)
max_Temp
min_Temp = df_final['temp'].quantile(0.001)
min_Temp
df2 = df_final[df_final['temp'] > min_Temp]
Temp_avg = df2['temp'].mean().round(2)
df_final.loc[df_final['temp'].isna(), 'temp'] = Temp_avg

### PRES
max_pres = df_final['pres'].quantile(0.999)
max_pres
min_pres = df_final['pres'].quantile(0.001)
min_pres
df2 = df_final[df_final['pres'] > min_pres]
pres_avg = df2['pres'].mean().round(2)
df_final.loc[df_final['pres'].isna(), 'pres'] = pres_avg

currentdate = datetime.now()
currentdate = currentdate.strftime("%Y-%m-%d")
datecheck = datetime.strptime(currentdate, "%Y-%m-%d") - pd.DateOffset(months=3)
datecheck = datecheck.strftime("%Y-%m-%d")

critical_cols = [
    "temp","tmin","tmax","txmn","txmx","prcp","pres",
    "name","country","region","latitude","longitude"
]
critical_cols_2 = ["ICEN"]

df_recent = df_final[df_final.index >= datecheck]
df_recent_clean = df_recent.dropna(subset=critical_cols)
df_recent_clean = df_recent.dropna(subset=critical_cols_2)

df_finalcleaned = pd.concat([
    df_final[df_final.index < datecheck],   # histórico intacto
    df_recent_clean                         # reciente limpio
])

## salvar en .csv
whichfields = ['temp','tmin','tmax','prcp','pres','name','country','latitude','longitude']
# Define csv filename
csvfile = 'station' + station_id + '.csv'

df_finalcleaned.to_csv(csvfile)
print(f"Guardado: {csvfile}")
