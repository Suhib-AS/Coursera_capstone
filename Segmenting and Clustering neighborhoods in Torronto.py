#!/usr/bin/env python
# coding: utf-8

# # Scraping wikipedia to create the dataframe

# In[66]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from geopy.geocoders import Nominatim

import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[2]:


response = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")


# In[3]:


soup = BeautifulSoup(response.text, 'html.parser')


# In[4]:


table = soup.find_all("table", class_="wikitable")


# In[5]:


data_heading = []
for h in table:
    th = h.find_all("th")
    for j in th:
        data_heading.append(j.string.replace("\n",""))
        
    


# In[6]:


data_heading


# In[7]:


data = []
for d in table:
    td = d.find_all("td")
    for cnt in td:
        data.append(cnt.string.replace("\n",""))


# In[ ]:





# In[8]:


dic = {}
borough_col = []
neighbourhood_col = []

for h in range(3):
    if h == 0:
        dic[data_heading[h]] = [i for i in data if data.index(i) in (list(range(0, len(data), 3)))]
    if h == 1:
        borough_col = []
        for counter in range(1,len(data),3):
            borough_col.append(data[counter])
                
        dic[data_heading[h]] = borough_col
            
    if h == 2:
        for co in range(2, len(data), 3):
            neighbourhood_col.append(data[co])
        dic[data_heading[h]] = neighbourhood_col
         

    


# In[ ]:





# In[ ]:





# In[9]:


df = pd.DataFrame(dic)


# In[10]:


df.head()


# In[11]:


df = df[df.Borough != "Not assigned"]


# In[12]:


df.head()


# In[13]:


df.shape


# ## Getting the coordinated for each postal code from the CSV file

# In[14]:


df_geo = pd.read_csv("http://cocl.us/Geospatial_data")


# In[15]:


df_geo.head()


# In[16]:


df_merged = pd.merge(df, df_geo,on="Postal Code", how="left")


# In[17]:


df_merged


# In[18]:


df_toronto = df_merged[df_merged["Borough"].str.contains("Toronto")]


# In[19]:


df_toronto.index = list(range(0,39))
df_toronto


# In[20]:


#to get the latitude and longtitude of Toronto
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="toronto")
location = geolocator.geocode("Toronto ,Ontario")
lat = location.latitude
long = location.longitude
print(f"Toronto coordinates are {lat} , {long}")


# In[21]:


try:
    import folium
except:
    get_ipython().system('pip install folium')


# In[22]:


map = folium.Map(location=[lat, long], zoom_start=11)


# In[23]:


df_merged.columns


# In[24]:


for latit, longit,borough, neighbor in zip(df_merged["Latitude"], df_merged["Longitude"], df_merged["Borough"], df_merged["Neighbourhood"]):
    label = f"{neighbor}, {borough}"
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker([latit, longit], radius=5, popup=label, color="red", fill=True, fill_color = "blue", fill_opacity = 0.7, parse_html=False).add_to(map)
map


# In[25]:


#using foursquare
client_id_ = "R0AEK44FOBRQJ2GUXYN4D1DPSTRX4JTC5Q24W2OCHEDNUK5L"
client_secret_ = "ILVPYCAIK0HPTBWLSC2JAE4CVNV5PDTZRWL4PYZGIFKVFQTQ"
version_ = "20180605"
limit_ = 100




# In[26]:


neighbourhood_name = df_toronto.loc[0,"Neighbourhood"]
neighborhood_lat = df_toronto.loc[0, "Latitude"]
neighborhood_Lon = df_toronto.loc[0, "Longitude"]


# In[27]:


params = dict(
client_id=client_id_,
client_secret=client_secret_,
v=version_,
ll = f"{format(neighborhood_lat, '.2f')}, {format(neighborhood_Lon, '.2f')}",
radius = 500,
)
url = r"https://api.foursquare.com/v2/venues/explore?"

resp = requests.get(url = url, params= params)
data = json.loads(resp.text)
data


# In[28]:


venues = data['response']['groups'][0]['items']


# In[29]:


nearby_venues = pd.json_normalize(venues)


# In[30]:


filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']


# In[31]:


nearby_venues = nearby_venues.loc[:,filtered_columns]


# In[ ]:





# In[32]:


nearby_venues['venue.categories'] = nearby_venues.apply(lambda x: x['venue.categories'][0]['name'], axis=1)


# In[ ]:





# In[33]:


# the statement below will loop through the dataframe cloumns and remove the word "venue", for example "venue.lng" will become "lng", and the assign it to the dataframe columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
nearby_venues


# In[34]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            client_id_, 
            client_secret_, 
            version_, 
            lat, 
            lng, 
            radius, 
            limit_)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[35]:


toronto_venues = getNearbyVenues(names=df_toronto['Neighbourhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude']
                                  )


# In[36]:


toronto_venues.head()
toronto_venues.groupby('Neighborhood').count()


# In[37]:


len(toronto_venues["Venue Category"].unique())


# ## Analyze Each Neighborhood

# In[38]:


toronto_onehot = pd.get_dummies(toronto_venues[["Venue Category"]], prefix="", prefix_sep="")
del toronto_onehot["Neighborhood"]


# In[39]:


try:
    toronto_onehot.insert(0,"Neighborhood",toronto_venues["Neighborhood"])
except:
    print("Neigborhood column already exist")


# In[40]:


toronto_grouped = toronto_onehot.groupby("Neighborhood").mean().reset_index()
toronto_grouped.head()


# In[41]:


num_top_venues = 5

for hood in toronto_grouped["Neighborhood"]:
    
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[42]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[47]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[70]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)


# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[62]:


# add clustering labels
#neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_toronto

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# In[68]:


map_clusters = folium.Map(location=[lat, long], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Examine clusters

# Cluster 1

# In[71]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# Cluster 2

# In[73]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# Cluster 3

# In[74]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# Cluster 4

# In[75]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# Cluster 5

# In[76]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




