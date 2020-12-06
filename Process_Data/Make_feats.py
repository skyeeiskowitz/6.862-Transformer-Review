import os
import glob
import pandas as pd
import numpy as np
import pickle

#This function gets returns a dictionary of dictionaries.
# Each dictionary is for one power plant and gives the key as each weather stations and the value as the distance to that station
def get_distance(solar_loc,weather_loc):
    from numpy import linalg as LA
    a = np.array(solar_loc)
    dist = {}
    for key in weather_loc:
        b = weather_loc[key]
        bb = b.split(',')
        b = [float(i) for i in bb]
        dist[key] = LA.norm(a - b)
    dist={k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    return dist

##Weather:
#Here, we determine the location of the stations
#Previously cleaned visibility data for each station is imported here so we know which stations to discard
df_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','clean_stations.pkl'))
keep = pd.read_pickle(df_path)

weather_data = os.path.join(os.path.dirname(__file__),'..','Data','weather_encoding.csv')
weather=pd.read_csv(weather_data)
weather['time']=pd.to_datetime(weather['valid'])
weather['lon'] = weather['lon'].astype(str)
weather['lat'] = weather['lat'].astype(str)
weather['location']=weather['lat'].str.cat(weather['lon'],sep=",")
stations=weather.station.unique()
keep=list(keep.columns)
stations=list(set(keep) & set(stations))

weather_loc=weather.set_index('station').to_dict()['location']
weather_loc = { your_key: weather_loc[your_key] for your_key in stations } #Dictionary of weather station coordinates

##Solar
solar_data = os.path.join(os.path.dirname(__file__),'..','Data','solar.csv')
solar=pd.read_csv(solar_data)
solar['time']=pd.to_datetime(solar['LocalTime'])

#To get solar data coordinates, need to parse the file names
current_dir = os.path.join(os.path.dirname(__file__),'al-pv-2006')
os.chdir(current_dir)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
current_dir = os.path.join(os.path.dirname(__file__))
os.chdir(current_dir)

a=all_filenames[0:137]
solar_loc=[]
for i in range(len(a)):
    c = []
    b=a[i].rsplit("_")
    c.append((float(b[1]),float(b[2])))
    solar_loc.append(c)

#Fill a list of dictionaries of the closest stations in ascending order
distance=[]
for i in range(len(solar_loc)):  # For each solar power plant
    distance.append(get_distance(solar_loc[i], weather_loc))

closest=[]
closest_val=[]

for sol in distance:
    closest.append(next(iter(sol))) #Cloesest weather station for each solar plant
    closest_val.append(next(iter(sol.values()))) #How closee the closest station is

closest2 = []
closest2_val = []
for sol in distance:
    be = pd.DataFrame.from_dict(sol, orient='index')
    closest2.append(be.index[1])
    closest2_val.append(be.iloc[1].values[0])

closest3=[]
closest3_val = []
for sol in distance:
   be = pd.DataFrame.from_dict(sol, orient='index')
   closest3.append(be.index[2])
   closest3_val.append(be.iloc[2].values[0])

with open('closest_val.pkl', 'wb') as f:
    pickle.dump(closest_val, f)

with open('closest2_val.pkl', 'wb') as f:
    pickle.dump(closest2_val, f)
    
with open('closest3_val.pkl', 'wb') as f:
    pickle.dump(closest3_val, f)



# closest4=[]
# for sol in distance:
#     b=pd.DataFrame.from_dict(sol,orient='index')
#     closest4.append(b.index[3])
    
# close3 = np.zeros(())
# for sol in distance:
#     df= pd.DataFrame.from_dict(sol, orient = 'index')
#     val = 0
#     for i =0:3:
#         val += vsby[df.index[i]]/df.index[i].values()
#     close3

import collections
counter=collections.Counter(closest)
histogram=counter.most_common()

closest_val = [i * 69.2 for i in closest_val]


###MAKE FEATURES

def fix_cloud(data):
    clouds = {'CLR': 0, 'FEW' : .25,  'SCT': .5,  'BKN' : 0.75, 'OVC' : 1, '   ' : np.nan}
    new_data = data.copy()
    for key in clouds:
        new_data = new_data.replace(key,clouds[key])
    
    new_data['skyc1'] = pd.to_numeric(new_data['skyc1'])
    return new_data
    

def process_weather(input_file):
    
    original_data = pd.read_csv(input_file)
    data_col = original_data.columns[-1]
    if data_col == 'skyc1':
        original_data = fix_cloud(original_data)
        
    df = original_data.sort_values(by=['station', 'valid'])
    df = df.set_index((['valid']))
    df.index = pd.to_datetime(df.index)
    new_df = pd.DataFrame()
    grouped = df.groupby(['station'])
    
    
    for station, data in grouped:
        ticks = data.loc[:,[data_col]]
        volumes = data[data_col].resample('1H').mean()
        new_df['station'] = volumes.interpolate('linear')
        new_df = new_df.rename(columns={'station': station})
        
    new_df.fillna(method='ffill')
    return new_df

tempf = process_weather(os.path.join(os.path.dirname(__file__),'..','Data','13_Weather_TempF.txt'))
vsby = process_weather(os.path.join(os.path.dirname(__file__),'..','Data','13_Weather_Vsby.txt'))
skyc1 = process_weather(os.path.join(os.path.dirname(__file__),'..','Data','13_Weather_Cloud.csv'))

out_path = os.path.join(os.path.dirname(__file__),'..','Data','Features')
tempf.to_csv(os.path.join(out_path,'tempf.csv'))
vsby.to_csv(os.path.join(out_path,'vsby.csv'))
skyc1.to_csv(os.path.join(out_path,'skyc1.csv'))


