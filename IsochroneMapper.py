'''
Public transport isochrone mapper for HSL API.
This code creates a HTML page that depicts minimum travel time between surrounding region and target location (aka isochrone map).
This is achieved by brute-force polling of given grid area using a predefined travel parameters. Results are then plotted
as contour regions and points.
The code was made for Helsinki region using HSL API, but you could apply it other regions as well with some modifications.

In order to make unrestricted API calls, start docker image
docker run --rm --name otp-hsl -p 9080:8080 -e ROUTER_NAME=hsl -e JAVA_OPTS=-Xmx5g -e ROUTER_DATA_CONTAINER_URL=https://api.digitransit.fi/routing-data/v2/hsl hsldevcom/opentripplanner
and use URL http://localhost:9080/otp/routers/hsl/index/graphql

Either way, polling is slow with speed around ~1.3/sec

-Janne Kauttonen

version history
16.2.2020 first version
30.8.2020 added polygon mapping with separate GUI tool
'''

import numpy as np
import os
import pickle
import requests
import geopy.distance
import time
from scipy.spatial import ConvexHull
import folium
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as plt_cm
import random
import pandas
import branca
from scipy.interpolate import griddata
import geojsoncontour
from shapely.geometry import Polygon, Point
import math
from scipy.ndimage.filters import gaussian_filter

## PARAMETERS - change these according to your taste and API service ##############################################
API_URL = r"https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql" # internet
API_URL = r"http://localhost:9080/otp/routers/hsl/index/graphql" # local
MAX_REQUEST_PER_SEC = 500 # careful not to get your IP banned for too fast requests (if online service)

# waltti regions	https://api.digitransit.fi/routing/v1/routers/waltti/index/graphql
# Entire Finland	https://api.digitransit.fi/routing/v1/routers/finland/index/graphql

RESULTFILE = "IsochroneMapper_results.pickle" # save results
TIME_RANGES = [20,30,40,50,60,70,80] # in minutes, routes beyond these are not plotted with contours
# MAPPING_AREA = {"TYPE":"RECT","CORNERS":(
#     (60.283088, 24.489482), # left top
#     (60.100719, 25.195741), # right bottom
# )} # X increases towards left, Y increases towards down
# MAPPING_AREA = {"TYPE":"CIRCLE","DISTANCE": 30} # distance in km
MAPPING_AREA = {"TYPE":"POLYGON","FILE": "HKI_polygon_coordinates.txt"} # distance in km
MAX_POINTS = 40000 # maximum number of polling points (randomized), Note: 20k takes few hours
OPACITY = 0.5 # layer opacity value for contours
MAPPING_ACCURACY_MIN = 0.200 # km, rounded down (smaller = more grid points)
TARGET_COORDINATE = (60.201672, 24.934188) # poll distance from here (e.g., your home or work)
# transport parameters for polling, defined by API service
TRANSPORT_PARAMETERS = {
    "date": "2020-09-09",
    "time": "08:00:00",
    "numItineraries": 3,
    "transportModes": ["RAIL","TRAM","WALK","BUS","SUBWAY"], # {"BUS","RAIL":True,"TRAM":True,"FERRY":False,"WALK":True,"SUBWAY":True}
    "walkReluctance": 1.0, # walk duration multiplier
    "walkBoardCost": 2*60, # sec
    "minTransferTime": 2*60, # sec
    "walkSpeed": 1.60, # 1.33 m/s
    "maxWalkDistance": 2000, # m
    "maxTransfers": 1,
    "ignoreRealtimeUpdates":True,
}
USE_OLD_DATA = False
isLegalPoint = lambda x,y: True  # glocal function handle (to be updated later)

# NOTE: below is an optional area-limiting function that you can use to limit valid points and save some time in polling.
# For most simple case, can be a separating line as below
def line_constraint(x):
    x1,y1=60.186486, 25.330735
    x2,y2=60.092571, 24.670135
    if x[1] < y2 + ((y2-y1)/(x2-x1))*(x[0]-x2):
        return True
    return False

#############################################################################################################

# conversion to km
def degree_to_km(x,y):
    return [110.574,111.320*math.cos(math.radians(x))]

def km_to_degree(x,y):
    xx,yy = degree_to_km(TARGET_COORDINATE[0],TARGET_COORDINATE[1])
    return [x/xx,y/yy]

# get number of steps for a rectangle
def get_stepcount(rect):
    x_dist = geopy.distance.geodesic(rect[0],rect[1]).km
    y_dist = geopy.distance.geodesic(rect[0],rect[3]).km
    x_dist_step = int(np.ceil(x_dist/MAPPING_ACCURACY_MIN))
    y_dist_step = int(np.ceil(y_dist/MAPPING_ACCURACY_MIN))
    x_step_dist = (rect[0][0]-rect[1][0])/x_dist_step
    y_step_dist = (rect[3][1]-rect[0][1])/y_dist_step
    return x_step_dist,x_dist_step,y_step_dist,y_dist_step

# sample points within polygon
def polygon_points(mypoints):
    global isLegalPoint
    poly = Polygon(mypoints)
    def random_points_within(poly, num_points):
        min_x, min_y, max_x, max_y = poly.bounds
        points = []
        while len(points) < num_points:
            random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
            if (random_point.within(poly)):
                points.append((random_point.coords.xy[0][0],random_point.coords.xy[1][0]))
        return points
    points = random_points_within(poly, MAX_POINTS)
    isLegalPoint = lambda x,y:Point([x,y]).within(poly) # update handle
    return points

# compute points
def compute_coordinates(mapping_dict):
    global isLegalPoint
    np.random.seed(1)
    random.seed(1)
    points, distance_from_target = [], []
    if mapping_dict["TYPE"]=="POLYGON":
        if "FILE" in mapping_dict:
            coords = pandas.read_csv(mapping_dict["FILE"],names=["LAT","LON"]).values.tolist()
        else:
            coords = mapping_dict["POINTS"]
        points = polygon_points(coords)
        for p in points:
            distance_from_target.append(geopy.distance.geodesic(p, TARGET_COORDINATE).km)
        return points, distance_from_target
    elif mapping_dict["TYPE"]=="CIRCLE":
        dtokm = degree_to_km(TARGET_COORDINATE[0],TARGET_COORDINATE[1])
        rect_corners=(
            (TARGET_COORDINATE[0]+mapping_dict["DISTANCE"]/dtokm[0],TARGET_COORDINATE[1]-mapping_dict["DISTANCE"]/dtokm[1]),
            (TARGET_COORDINATE[0]-mapping_dict["DISTANCE"]/dtokm[0],TARGET_COORDINATE[1]+mapping_dict["DISTANCE"]/dtokm[1])
        )
    elif mapping_dict["TYPE"]=="RECT":
        rect_corners=mapping_dict["CORNERS"]

    rect = (rect_corners[0],
            (rect_corners[1][0],rect_corners[0][1]),
            rect_corners[1],
            (rect_corners[0][0],rect_corners[1][1]),)
    x_step_dist, x_dist_step, y_step_dist, y_dist_step = get_stepcount(rect)
    for x in range(x_dist_step):
        for y in range(y_dist_step):
            new_point = (rect[1][0] + x * x_step_dist,rect[0][1] + y * y_step_dist)
            if line_constraint(new_point):
                if mapping_dict["TYPE"]=="CIRCLE":
                    dist = geopy.distance.geodesic(new_point,TARGET_COORDINATE).km
                    if dist > MAPPING_AREA["DISTANCE"]:
                        continue
                points.append(new_point)
                distance_from_target.append(geopy.distance.geodesic(points[-1],TARGET_COORDINATE).km)
    return points,distance_from_target

# api query
def API_caller(query): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.post(API_URL, json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

# helper function to format transport modes
def make_transportModes(modes):
    return ", ".join(["{mode: %s}" % x for x in modes]) # "[{mode: BUS}, {mode: RAIL}, {mode: TRAM}, {mode: FERRY}, {mode: WALK}]"

# GraphQL query template for HSL api
query = """
{{
  plan(
    fromPlace: "{fromPlace}",
    toPlace: "{toPlace}",
    date: "{date}",
    time: "{time}",
    numItineraries: {numItineraries},
    transportModes: [{transportModes}]
    walkReluctance: {walkReluctance},
    walkBoardCost: {walkBoardCost},
    minTransferTime: {minTransferTime},
    walkSpeed: {walkSpeed},
    maxTransfers: {maxTransfers},
  ) {{
    itineraries{{
      walkDistance
      duration
      legs {{
        mode
        startTime
        endTime
        from {{
          lat
          lon
          name
        }}
        to {{
          lat
          lon
          name
        }}
        trip {{
          tripHeadsign
          routeShortName
        }}
        distance
        legGeometry {{
          length
        }}
      }}
    }}
  }}
}}
"""

# plot convex hull
def create_convexhull_polygon(map_object, list_of_points, layer_name, line_color, fill_color, weight, text):
    # Since it is pointless to draw a convex hull polygon around less than 3 points check len of input
    if len(list_of_points) < 3:
        return

    # Create the convex hull using scipy.spatial
    form = [list_of_points[i] for i in ConvexHull(list_of_points).vertices]

    # Create feature group, add the polygon and add the feature group to the map
    fg = folium.FeatureGroup(name=layer_name)
    fg.add_child(folium.vector_layers.Polygon(locations=form, color=line_color, fill_color=fill_color,weight=weight, popup=(folium.Popup(text))))
    map_object.add_child(fg)

    return (map_object)

# Function to draw points in the map
def draw_points(map_object, list_of_points, layer_name=None,color=None,text=None,radius=1,show=True):
    fg = folium.FeatureGroup(name=layer_name,show=show)
    islist=False
    if isinstance(color,list):
        islist = True
    for i,point in enumerate(list_of_points):
        if islist:
            fg.add_child(folium.Circle(point, radius=radius, color=color[i], fill_color=color[i],popup=(folium.Popup(text[i]))))
        else:
            fg.add_child(folium.CircleMarker(point, radius=radius, color=color, fill_color=color))
    map_object.add_child(fg)

# route as html string
def parse_route(route):
    r = ""
    tot = 0
    for i in range(len(route["legs"])):
        if i>0:
            r+="<br>"
        x = (route["legs"][i]["endTime"] - route["legs"][i]["startTime"]) / 60 / 1000
        r += "%s (%.1fmin)" % (route["legs"][i]["mode"], x)
        tot += x
    start = "(%.4f, %.4f)" % (route["legs"][0]["from"]["lat"],route["legs"][0]["from"]["lon"])
    r = "<h3><strong>%.1fmin</strong> (travel %.1fmin)<br>%s:</h3><p>%s</p>" % (route['duration'] / 60, tot,start,r)
    return r

# return time group of the results for one point
def get_group(result):
    times = []
    for i in range(len(result)):
        times.append(result[i]['duration'])
    m_ind = int(np.argmin(times)) # minutes, fastest route for this point
    m=times[m_ind]/60.0
    info = parse_route(result[m_ind])
    for i in range(len(TIME_RANGES)):
        if m<TIME_RANGES[i]:
            return i,m,info
    return i+1,np.nan,info

def get_color_index(val,borders):
    for i in range(len(borders)-1):
        if borders[i]<=val<borders[i+1]:
            return i
    return i

# get colormap
def get_colormap(z_orig):
    # Setup colormap
    color_list = plt.cm.get_cmap(plt_cm.autumn)(np.linspace(0,1,len(TIME_RANGES))) #
    color_list = [colors.to_hex(x, keep_alpha=False) for x in color_list]
    vmin = np.nanmin(z_orig)
    vmax = np.nanmax(z_orig)
    levels = len(color_list)
    cm = branca.colormap.LinearColormap(color_list, vmin=vmin, vmax=vmax).to_step(levels)

    # Setup colormap
    N=30
    TIME_RANGES_dense = np.linspace(0,TIME_RANGES[-1],N+1)
    color_list = plt.cm.get_cmap(plt_cm.autumn)(np.linspace(0,1,N)) #
    color_list = [colors.to_hex(x, keep_alpha=False) for x in color_list]
    return cm,color_list,TIME_RANGES_dense

# add contour to map
def add_contour(x_orig,y_orig,z_orig,my_map_global,cm,layer_name,threshold=None,add_colormap=True,show = True,smooth_sigma_km=0):

    # The original data
    x_orig = np.array(x_orig)
    y_orig = np.array(y_orig)
    z_orig = np.array(z_orig)

    # Make a grid
    rect_corners = [(np.max(x_orig), np.min(y_orig)),(np.min(x_orig), np.max(y_orig))]
    rect = (rect_corners[0],
            (rect_corners[1][0],rect_corners[0][1]),
            rect_corners[1],
            (rect_corners[0][0],rect_corners[1][1]),)
    x_step_dist, x_dist_step, y_step_dist, y_dist_step = get_stepcount(rect)

    x_arr = np.linspace(np.min(y_orig), np.max(y_orig), max(200,min(800,y_dist_step*4)))
    y_arr = np.linspace(np.min(x_orig), np.max(x_orig), max(200,min(800,x_dist_step*4)))
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

    def nullify_outside(z,x,y):
        for col in range(x.shape[1]):
            for row in range(y.shape[0]):
                if not isLegalPoint(y[row,col],x[row,col]):
                    z[row,col] = np.nan

    # Grid the values
    z_mesh = griddata((y_orig, x_orig), z_orig, (x_mesh, y_mesh), method='linear')

    nullify_outside(z_mesh,x_mesh,y_mesh)

    non_zero_count1 = np.sum(~np.isnan(z_mesh))
    # Gaussian filter the grid to make it smoother
    if smooth_sigma_km>0:
        sigma = km_to_degree(smooth_sigma_km,smooth_sigma_km)
        sigma[1] = sigma[1]/ x_step_dist
        sigma[0] = sigma[0]/ y_step_dist
        z_mesh = gaussian_filter(z_mesh, sigma, mode='constant')
        nullify_outside(z_mesh, x_mesh, y_mesh)

    if threshold!=None:
        z_mesh[z_mesh>threshold]=np.nan

    non_zero_count2 = np.sum(~np.isnan(z_mesh))
    print("... non-nan counts %i --> %i" % (non_zero_count1,non_zero_count2))

    # Create the contour
    levels = len(cm.colors)
    vmin = cm.vmin
    vmax = cm.vmax
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh,[0]+TIME_RANGES,colors=cm.colors, vmin=vmin,vmax=vmax)

    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        #min_angle_deg=3.0,
        #ndigits=5,
        stroke_width=2)

    fg = folium.FeatureGroup(name=layer_name,show=show)
    # Plot the contour plot on folium
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            'color': x['properties']['stroke'],
            'weight': x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'fillOpacity': OPACITY
        }).add_to(fg)  # geomap

    # Add the colormap to the folium map
    cm.caption = layer_name
    # cm.add_to(fg)
    if add_colormap:
        my_map_global.add_child(cm)
    my_map_global.add_child(fg)

    return my_map_global

if __name__ == "__main__":
    TRANSPORT_PARAMETERS["transportModes"] = make_transportModes(TRANSPORT_PARAMETERS["transportModes"])
    REQUEST_INTERVAL_WAIT = 1.0/MAX_REQUEST_PER_SEC

    print("computing points")
    points,distance_from_target = compute_coordinates(MAPPING_AREA)

    # shuffle point order
    random.seed(1)
    random.shuffle(points)
    points=points[0:min(len(points),MAX_POINTS)]

    print("total %i points to test (takes at least %.2f min)" % (
    len(points), (REQUEST_INTERVAL_WAIT * len(points) / 60.0)))

    # load old results
    results = []
    if os.path.isfile(RESULTFILE):
        old_results, old_points, old_TRANSPORT_PARAMETERS = pickle.load(open(RESULTFILE, "rb"))
        if (old_TRANSPORT_PARAMETERS==TRANSPORT_PARAMETERS and points==old_points) or USE_OLD_DATA:
            results = old_results
            points =  old_points
            print("old data loaded, already have %i of %i points" % (len(results),len(points)))

    # start polling
    print("starting polling")
    params = dict(TRANSPORT_PARAMETERS)
    start = time.time()
    start_init = start
    lastsave = start
    count = 0
    old_points = len(results)
    for i in range(len(results),len(points)):
        # run your code
        params["fromPlace"] = str(points[i])[1:-1]
        params["toPlace"] = str(TARGET_COORDINATE)[1:-1]

        # sleep to avoid too fast polling
        end = time.time()
        elapsed = end - start  # seconds
        time.sleep(max(0,REQUEST_INTERVAL_WAIT-elapsed))
        start = time.time()

        s = query.format(**params)
        RES = API_caller(s)  # Execute the query
        result = RES["data"]["plan"]["itineraries"]
        results.append(result)
        if len(result)>0:
            count+=1
        if (end - lastsave)>30 or i==len(points)-1:
            # save results every 30s
            lastsave = start
            print("point %i of %i: (%s, distance %.3fkm): found %i routes (data ratio %.3f, %.2fpoints/sec)" % (
            i + 1, len(points), params["fromPlace"], distance_from_target[i], len(result),(count/i),(i+1-old_points)/(end-start_init)))
            pickle.dump((results,points,TRANSPORT_PARAMETERS),open(RESULTFILE,"wb"))

    # analyze results
    print("analyzing results")
    x,y,z,info = [],[],[],[]
    ind_tip = []
    for i in range(len(results)):
        x.append(points[i][0])
        y.append(points[i][1])
        if len(results[i])>0:
            k,t,s = get_group(results[i])
            z.append(t)
            info.append(s)
            ind_tip.append(i)
        else:
            z.append(np.nan)

    # get global color maps for contours (first) and points (two latter ones)
    cm,color_list,TIME_RANGES_dense = get_colormap(z)

    # create one map for each smoothing factor
    for smooth_km in [0,0.05,0.1,0.5]:

        #flatten_list = lambda l: [item for sublist in l for item in sublist]
        # Initialize map
        my_map_global = folium.Map(location=TARGET_COORDINATE, zoom_start=10)

        print("drawing layers")
        # add combined contour
        print("..full contour")
        my_map_global = add_contour(x, y, z, my_map_global,cm,"Full_contour",smooth_sigma_km=smooth_km)

        # plot individual time layers
        for i in range(len(TIME_RANGES)):
            #color = colors.to_hex(color_list[i],keep_alpha=False)
            s = "Max_%imin" % (TIME_RANGES[i])
            print("..layer %s" % s)
            #create_convexhull_polygon(my_map_global,pointlist[i],layer_name=s,line_color=None, fill_color=color, weight=5,text=s)
            my_map_global = add_contour(x, y, z, my_map_global,cm,s,threshold=TIME_RANGES[i],add_colormap=False,show=False)

        # plot point layers
        print("drawing points")
        print("..layer poll points")
        draw_points(my_map_global,points, layer_name='Poll_points',color='royalblue',show=False)
        print("..layer target")
        draw_points(my_map_global,[TARGET_COORDINATE], layer_name='Target',color=['red'],text=["Target %s" % str(TARGET_COORDINATE)],radius=1)
        print("..layer route points")
        draw_points(my_map_global, [points[i] for i in ind_tip], layer_name='Points_with_route',
                    color=[color_list[get_color_index(z[i], TIME_RANGES_dense)] for i in ind_tip], text=info, show=False,
                    radius=20)

        folium.LayerControl(collapsed=False).add_to(my_map_global)

        print("saving the map")
        if smooth_km>0:
            my_map_global.save("mapper_results_smoothed_%0.3fkm.html" % smooth_km)
        else:
            my_map_global.save("mapper_results.html")

    print("All done!")