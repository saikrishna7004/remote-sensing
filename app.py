from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64

import sys
import datacube
import matplotlib.pyplot as plt
from datacube.utils.cog import write_cog
from dea_tools.plotting import rgb

dc = datacube.Datacube(app="Flask_Text")

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method=="POST":
        print(dict(request.form))

        central_lat = float(request.form['latitude'])
        central_lon = float(request.form['longitude'])

        buffer = float(request.form['buffer'])

        study_area_lat = (central_lat - buffer, central_lat + buffer)
        study_area_lon = (central_lon - buffer, central_lon + buffer)

        time_range = (request.form['fromdate'], request.form['todate'])

        # Connect to the datacube
        dc = datacube.Datacube(app='vegetation_change_analysis')

        # Load the Sentinel-2 data for the study area and time range
        ds = dc.load(product='s2a_sen2cor_granule',
            x=study_area_lon,
            y=study_area_lat,
            time=time_range,
            measurements=['nir', 'red'],
            output_crs='EPSG:4326',
            resolution=(-0.00027, 0.00027)
        )

        # Compute the NDVI
        ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)

        # Compute the mean NDVI for each year
        ndvi_2022 = ndvi.sel(time='2022').mean(dim='time')
        ndvi_2023 = ndvi.sel(time='2023').mean(dim='time')

        # Compute the difference in mean NDVI between the two years
        ndvi_diff = ndvi_2023 - ndvi_2022

        # Plot the NDVI difference
        plt.figure(figsize=(10, 5))
        plt.imshow(ndvi_diff, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
        plt.colorbar()
        plt.title('Vegetation Change 2022-2023')
        
        img = io.BytesIO()
        plt.savefig(img, format = 'png')
        img.seek(0)
        plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
        
        return render_template('index.html', plot_url=plot_data,  data=str(dict(request.form)), lat=central_lat, lon=central_lon)
    
    return render_template("index.html", lat=15.733048, lon=80.945512)

if __name__ == '__main__':
    app.run(debug=True)