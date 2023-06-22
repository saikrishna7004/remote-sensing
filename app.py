import json
from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64
import datacube
from datacube.utils.geometry import CRS
from pyproj import Transformer

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
        
        coordinates = data['coordinates']
        time_range = (data['fromdate'], data['todate'])
        study_area_lat = (coordinates[0][0], coordinates[1][0])
        study_area_lon = (coordinates[1][1], coordinates[2][1])

        try:
            dc = datacube.Datacube(app='water_change_analysis')

            ds = dc.load(product='s2a_sen2cor_granule',
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['B04_10m', 'B03_10m', 'B02_10m', 'B08_10m'],
                output_crs='EPSG:6933',
                resolution=(-30, 30)
            )

            if analysis_type=="ndvi":
                res = (ds.B08_10m - ds.B04_10m) / (ds.B08_10m + ds.B04_10m)
            elif analysis_type=="ndwi":
                res = (ds.B03_10m - ds.B08_10m) / (ds.B03_10m + ds.B08_10m)
            elif analysis_type=="forest":
                ndvi = (ds.B08_10m - ds.B04_10m) / (ds.B08_10m + ds.B04_10m)
                evi = 2.5 * ((ds.B08_10m - ds.B04_10m) / (ds.B08_10m + 6 * ds.B04_10m - 7.5 * ds.B02_10m + 1))

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
                    print(dense_forest_mask)
                    # Calculate the forest cover area for each forest type
                    dense_forest_cover_area = np.sum(dense_forest_mask[i]) * pixel_area
                    open_forest_cover_area = np.sum(open_forest_mask[i]) * pixel_area
                    sparse_forest_cover_area = np.sum(sparse_forest_mask[i]) * pixel_area

                    print('areas', dense_forest_cover_area, open_forest_cover_area, sparse_forest_cover_area)

                    # Calculate the total forest cover area
                    total_forest_cover_area = dense_forest_cover_area + open_forest_cover_area + sparse_forest_cover_area

                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area
                    
                    print("1 data added")

                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                dense_forest_cover_area, open_forest_cover_area,
                                sparse_forest_cover_area, total_forest_cover_area, original])
                    
                df = pd.DataFrame(data[1:], columns=data[0])
                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')

                X = df[["year", "month"]]
                y = df["dense_forest"]
                y2 = df["open_forest"]
                y3 = df["sparse_forest"]

                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict([[2024, 5]])
                print(df, y_pred)
                rf_regressor2 = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor2.fit(X, y2)
                y_pred2 = rf_regressor2.predict([[2024, 5]])
                print(df, y_pred2)
                rf_regressor3 = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor3.fit(X, y3)
                y_pred3 = rf_regressor3.predict([[2024, 5]])
                print(df, y_pred3)

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')

                print("year-month done")

                plot_data = [
                    go.Scatter(
                        x = df['year-month'],
                        y = df['dense_forest']/1000000,
                        name = "Dense Actual"
                    ),
                    go.Scatter(
                        x = ['2024-05'],
                        y = y_pred/1000000,
                        name = "Dense Predicted"
                    ),
                    go.Scatter(
                        x = df['year-month'],
                        y = df['open_forest']/1000000,
                        name = "Open Actual"
                    ),
                    go.Scatter(
                        x = ['2024-05'],
                        y = y_pred2/1000000,
                        name = "Open Predicted"
                    ),
                    go.Scatter(
                        x = df['year-month'],
                        y = df['sparse_forest']/1000000,
                        name = "Sparse Actual"
                    ),
                    go.Scatter(
                        x = ['2024-05'],
                        y = y_pred3/1000000,
                        name = "Sparse Predicted"
                    ),
                ]

                print("Plot plotted")

                plot_layout = go.Layout(
                    title='Dense Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)

                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
                print(area_name)

                return jsonify({"plot": plot_json, "type": "Random Forest Analysis", "area_name": area_name})
            else:
                return jsonify({"error": "Invalid type"})

            res_start = res.sel(time=slice(time_range[0], time_range[1])).min(dim='time')
            res_end = res.sel(time=slice(time_range[0], time_range[1])).max(dim='time')
            res_diff = res_end - res_start

            if analysis_type=="ndvi":
                title = 'Vegetation'
                cmap = 'YlGn_r'
            elif analysis_type=="ndwi":
                title = 'Water'
                cmap = 'cividis'

            plt.figure(figsize=(10, 4))
            
            plot = res.plot(col='time', vmin=0, vmax=1, col_wrap=3, cmap=cmap)
            for ax, time in zip(plot.axes.flat, res.time.values):
                ax.set_title(str(time).split('T')[0])

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()
            
            area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
            print(area_name)

            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates, "type": analysis_type, "area_name": area_name})
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
    print(coordinates)
    return jsonify({'coordinates': coordinates})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')