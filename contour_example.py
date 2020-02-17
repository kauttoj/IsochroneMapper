#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import folium
import branca
from folium import plugins
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage
from matplotlib import colors
import matplotlib.cm as plt_cm

TARGET_COORDINATE = (60.201672, 24.934188) # distence from here

# Function to draw points in the map
def draw_points(map_object, list_of_points, layer_name, line_color, fill_color, text,radius=0.75,show=True,max_points=2000):

    fg = folium.FeatureGroup(name=layer_name,show=show)

    for point in list_of_points:
        fg.add_child(folium.CircleMarker(point, radius=radius, color=line_color, fill_color=fill_color,
                                         popup=(folium.Popup(text))))

    map_object.add_child(fg)

def make_contour(x_orig,y_orig,z_orig):

    # returns a map with the convex hull polygon from the points as a new layer

    # Setup
    temp_mean = 12
    temp_std = 2
    debug = False

    # Setup colormap
    color_list = plt.cm.get_cmap(plt_cm.autumn)(np.linspace(0,1,5))
    color_list = [colors.to_hex(x,keep_alpha=False) for x in color_list]
    vmin = temp_mean - 2 * temp_std
    vmax = temp_mean + 2 * temp_std
    levels = len(color_list)
    cm = branca.colormap.LinearColormap(color_list, vmin=vmin, vmax=vmax).to_step(levels)

    # Create a dataframe with fake data
    df = pd.DataFrame({
        'longitude': y_orig,
        'latitude': x_orig,
        'temperature': z_orig})

    # The original data
    x_orig = np.asarray(df.longitude.tolist())
    y_orig = np.asarray(df.latitude.tolist())
    z_orig = np.asarray(df.temperature.tolist())

    # Make a grid
    x_arr = np.linspace(np.min(x_orig), np.max(x_orig), 500)
    y_arr = np.linspace(np.min(y_orig), np.max(y_orig), 500)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

    # Grid the values
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')

    # Gaussian filter the grid to make it smoother
    sigma = [5, 5]
    z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')

    # Create the contour
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=color_list, linestyles='None', vmin=vmin,
                            vmax=vmax)

    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.5)

    return geojson,cm

# Set up the folium plot
geomap = folium.Map([TARGET_COORDINATE[0],TARGET_COORDINATE[1]], zoom_start=10, tiles="openstreetmap")

fg = folium.FeatureGroup(name="temparature_contour")
#fg.add_child(cm)
#geomap.add_child(fg)

y_orig = np.random.normal(TARGET_COORDINATE[1], 0.10, 1000)
x_orig = np.random.normal(TARGET_COORDINATE[0], 0.10, 1000)
z_orig = np.maximum(0, np.random.normal(20,5, 1000))
z_orig[z_orig<9]=np.nan

geojson,cm = make_contour(x_orig,y_orig,z_orig)

# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'color': x['properties']['stroke'],
        'weight': x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity': 0.50,
    }).add_to(fg) # geomap

# Add the colormap to the folium map
cm.caption = 'Temperature'
#cm.add_to(fg)
geomap.add_child(cm)
geomap.add_child(fg)

# Fullscreen mode
#plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)
draw_points(geomap,[(TARGET_COORDINATE[0],TARGET_COORDINATE[1])],"centerpoint","red","red","centerpoint")

folium.LayerControl(collapsed=False).add_to(geomap)

# Plot the data
geomap.save(f'folium_contour_temperature_map.html')