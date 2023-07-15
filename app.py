import json
from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64, xarray
import datacube
from datacube.utils.geometry import CRS
from pyproj import Transformer
import odc.algo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import plotly.graph_objs as go
import plotly.offline as pyoff
import plotly.io as pio

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import warnings
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

from geopy.geocoders import Nominatim

def get_area_name(latitude, longitude):
    geolocator = Nominatim(user_agent='my-app')  # Initialize the geocoder
    location = geolocator.reverse((latitude, longitude))  # Reverse geocode the coordinates
    if location is not None:
        address_components = location.raw['address']
        city_name = address_components.get('mandal', '')
        if not city_name:
            city_name = address_components.get('city', '')
        if not city_name:
            city_name = address_components.get('town', '')
        if not city_name:
            city_name = address_components.get('village', '')
        return city_name
    else:
        return "City name not found"

dc = datacube.Datacube(app="Flask_Text")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", lat=15.733048, lon=80.945512)

@app.route('/type/<analysis_type>', methods=['POST'])
def analysis(analysis_type):
    if request.method=="POST":
        data = request.get_json()
        
        time_range = (data['fromdate'], data['todate'])

        try:
            if analysis_type=="rainfall":
                
                dist_data = {
                    'Adilabad Rural': {'min_lon': 78.5229, 'min_lat': 19.6545, 'max_lon': 78.5709, 'max_lat': 19.6982},
                    'Gadiguda': {'min_lon': 78.7355476, 'min_lat': 19.317508, 'max_lon': 78.8155476, 'max_lat': 19.407508},
                }
                name = data['distName']
                rectangle = dist_data[name]

                data = pd.read_csv('pre_final.csv')

                mandal = name.split("mandal")[0]
                rainfall_df = data[(data['Mandal'] == mandal)]
                rainfall_df['Date'] = pd.to_datetime(rainfall_df['Date'])
                rainfall_df = rainfall_df.sort_values(by='Date')

                dc = datacube.Datacube(app="04_Plotting")
                lat_range = (rectangle['min_lat'], rectangle['max_lat'])
                lon_range = (rectangle['min_lon'], rectangle['max_lon'])
                ds = dc.load(
                    product=["s2a_sen2cor_granule","s2b_sen2cor_granule"],
                    measurements=["red","green","blue", "nir"],
                    x=lon_range,
                    y=lat_range,
                    time=time_range, 
                    output_crs='EPSG:6933',
                    resolution=(-30, 30)
                )

                ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
                evi = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))

                # Create forest masks based on NDVI and EVI thresholds
                dense_forest_mask = np.where((ndvi > 0.6) & (ndvi < 0.8) & (evi > 0.4), 1, 0)
                open_forest_mask = np.where((ndvi > 0.3) & (ndvi < 0.6) & (evi > 0.2) & (evi < 0.4), 1, 0)
                sparse_forest_mask = np.where((ndvi > 0.1) & (ndvi < 0.3) & (evi < 0.2), 1, 0)

                print("Thresholds applied")

                # Calculate the area of each pixel
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])
                print('pixel_area', pixel_area)

                data = [['day', 'month', 'year', 'dense_forest', 'open_forest', 'sparse_forest', 'forest', 'total']]

                for i in range(dense_forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    new_data_time = data_time.split("-")

                    dense_forest_cover_area = np.sum(dense_forest_mask[i]) * pixel_area
                    open_forest_cover_area = np.sum(open_forest_mask[i]) * pixel_area
                    sparse_forest_cover_area = np.sum(sparse_forest_mask[i]) * pixel_area

                    total_forest_cover_area = dense_forest_cover_area + open_forest_cover_area + sparse_forest_cover_area

                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area

                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                dense_forest_cover_area, open_forest_cover_area,
                                sparse_forest_cover_area, total_forest_cover_area, original])

                df = pd.DataFrame(data[1:], columns=data[0])
                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')

                df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])

                rainfall_df['year-month'] = rainfall_df['Date'].dt.to_period('M')
                rainfall_df['year-month'] = rainfall_df['year-month'].astype(str)
                rainfall_df = rainfall_df.groupby(['year-month'])['Rainfall (mm)'].mean().reset_index()

                plot_data = [
                    go.Scatter(
                        x = df['Date'],
                        y = df['forest']/1000000,
                        name = "Dense Forest",
                        yaxis="y1",
                        mode='lines+markers',
                    ),
                    go.Scatter(
                        x = rainfall_df['year-month'],
                        y = rainfall_df['Rainfall (mm)'],
                        name = "Rainfall (mm)",
                        yaxis="y2",
                        mode='lines+markers',
                    ),
                ]

                plot_layout = go.Layout(
                    title='Dense Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)
                fig.update_layout(
                    xaxis_title="Year-Month",
                    yaxis_title="Dense Forest Area (sq.km)"
                )
                fig.update_layout(
                    yaxis=dict(
                        title="Dense Forest Area (sq.km)",
                        side="left",
                        range=[0, df['forest'].max() / 1000000]  # Set the range for the left y-axis
                    ),
                    yaxis2=dict(
                        title="Rainfall (mm)",
                        side="right",
                        overlaying="y",
                        range=[0, rainfall_df['Rainfall (mm)'].max()]# Set the range for the right y-axis
                    ),
                    xaxis=dict(title="Year-Month"),
                )
                
                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                return jsonify({"plot": plot_json, "type": "Forest vs Rainfall", "area_name": name})
            
            coordinates = data['coordinates']
            study_area_lat = (coordinates[0][0], coordinates[1][0])
            study_area_lon = (coordinates[1][1], coordinates[2][1])

            dc = datacube.Datacube(app='water_change_analysis')

            ds = dc.load(product='s2a_sen2cor_granule',
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:4326',
                resolution=(-0.00027, 0.00027)
            )
            ds = odc.algo.to_f32(ds)

            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            elif analysis_type=="evi":
                res = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
                res = xarray.where(~np.isfinite(res), 0.0, res)
                print(res)
            elif analysis_type=="forest":
                ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
                evi = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))

                # Create forest masks based on NDVI and EVI thresholds
                forest_mask = np.where((ndvi > 0.5) & (evi > 0.2), 1, 0)

                # Calculate the area of each pixel
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])
                print('pixel_area', pixel_area)

                data = [['day', 'month', 'year', 'forest', 'total']]

                for i in range(forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    new_data_time = data_time.split("-")

                    # Calculate the forest cover area for each forest type
                    forest_cover_area = np.sum(forest_mask[i]) * pixel_area

                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area
                    
                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                forest_cover_area, original])
                
                df = pd.DataFrame(data[1:], columns=data[0])
                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')

                grouped_df = df.groupby(['year', 'month'])

                # Step 3: Calculate the mean of 'forest_field' for each group
                mean_forest_field = grouped_df['forest'].mean()

                # Step 4: Optional - Reset the index of the resulting DataFrame
                mean_forest_field = mean_forest_field.reset_index()
                print(mean_forest_field)

                df = mean_forest_field

                X = df[["year", "month"]]
                y = df["forest"]

                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict(X)
                print(df, y_pred)

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')

                print("year-month done")

                plot_data = [
                    go.Scatter(
                        x = df['year-month'],
                        y = df['forest']/1000000,
                        name = "Forest Actual"
                    ),
                    go.Scatter(
                        x = df['year-month'],
                        y = y_pred/1000000,
                        name = "Forest Predicted"
                    )
                ]

                print("Plot plotted")

                plot_layout = go.Layout(
                    title='Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)

                fig.update_layout(
                    xaxis_title="Year-Month",
                    yaxis_title="Forest Area (sq.km.)"
                )
                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
                print(area_name)

                return jsonify({"plot": plot_json, "type": "Random Forest Analysis", "area_name": area_name})
            else:
                return jsonify({"error": "Invalid type"})
            sub_res = res.isel(time=[0, -1])
            print(sub_res)

            mean_res = res.mean(dim=['latitude', 'longitude'], skipna=True)
            mean_res_rounded = np.array(list(map(lambda x: round(x, 4), mean_res.values.tolist())))
            mean_res_rounded = mean_res_rounded[np.logical_not(np.isnan(mean_res_rounded))]
            mean_res_rounded = [0 if (i>1 or i<-1) else i for i in mean_res_rounded]
            labels = list(map(lambda x: x.split('T')[0], [i for i in np.datetime_as_string(res.time.values).tolist()])) 

            plot = sub_res.plot(col='time', col_wrap=2, vmin=-1, vmax=1)
            for ax, time in zip(plot.axes.flat, sub_res.time.values):
                ax.set_title(str(time).split('T')[0])

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()
            
            try:
                area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
                if area_name=="":
                    area_name = "Unknown"
                print(area_name)
            except:
                area_name = "Unknown"

            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates, "type": analysis_type, "area_name": area_name, "mean_res_rounded": mean_res_rounded, "labels": labels})
        except Exception as e:
            return jsonify({"error": e})
    return jsonify({"error": "Invalid method: "+request.method})


@app.route('/datasets', methods=['GET'])
def datasets():
    dc = datacube.Datacube()

    product = 's2a_sen2cor_granule'

    # Get the available datasets for the specified product
    datasets = dc.find_datasets(product=product)

    # Initialize an empty list to store the coordinates
    coordinates = []

    # Iterate over the datasets and extract the coordinates
    for dataset in datasets:
        bounds = dataset.bounds
        coordinates.append([[bounds.left, bounds.bottom], [bounds.right, bounds.top]])

    # Print the coordinates
    for coord in coordinates:
        print(coord)
    # print(coordinates)
    return jsonify({'coordinates': coordinates})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
