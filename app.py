from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64, json

import sys
import datacube
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')
from datacube.utils.cog import write_cog
from dea_tools.plotting import rgb

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
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:4326',
                resolution=(-0.00027, 0.00027)
            )

            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            else:
                return jsonify({"error": "Invalid type"})

            res_start = res.sel(time='2022').mean(dim='time')
            res_end = res.sel(time='2023').mean(dim='time')
            res_diff = res_end - res_start

            if analysis_type=="ndvi":
                title = 'Vegetation'
                cmap = 'RdYlBu'
            elif analysis_type=="ndwi":
                title = 'Water'
                cmap = 'RdBu'

            plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(2, 2)

            plt.subplot(gs[0, 0])
            plt.imshow(res_start, cmap=cmap, vmin=-1, vmax=1)
            plt.title(title+' '+data['fromdate'][:4])

            plt.subplot(gs[0, 1])
            plt.imshow(res_end, cmap=cmap, vmin=-1, vmax=1)
            plt.title(title+' '+data['todate'][:4])

            plt.subplot(gs[1, :])
            plt.imshow(res_diff, cmap=cmap, vmin=-1, vmax=1)
            plt.title(title+' Change')

            plt.colorbar(shrink=0.5)

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()
            
            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates})
        except Exception as e:
            return jsonify({"error": e})
    return jsonify({"error": "Invalid method: "+request.method})

if __name__ == '__main__':
    app.run(debug=True)