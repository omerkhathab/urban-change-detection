from flask import Flask, render_template, request, jsonify
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import ee
import os
import time
import folium
import rasterio
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from change_detection import run_change_detection
from long_term import add_population_layer, calculate_population

ee.Authenticate()
ee.Initialize(project="urban-change-detection")

app = Flask(__name__)

status_message = ""

def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    
    gauth.SaveCredentialsFile("credentials.json")
    
    return GoogleDrive(gauth)

def wait_for_tasks(export_tasks):
    print("Waiting for export tasks to complete...")
    while any(t.status()['state'] in ['READY', 'RUNNING'] for t in export_tasks):
        time.sleep(10)
    
    failed = [t for t in export_tasks if t.status()['state'] == 'FAILED']
    if failed:
        raise RuntimeError("One or more of your export tasks failed.")

    print("All export tasks completed.")

def download_files_from_drive(startDate, endDate, drive, folder_name, local_folder):
    # Step 1: Find the folder ID by its name
    folder_list = drive.ListFile({
        'q': f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    
    if not folder_list:
        raise Exception(f"Folder '{folder_name}' not found in Drive or not shared with this service account.")

    folder_id = folder_list[0]['id']
    print(f"Found folder '{folder_name}' with ID: {folder_id}")

    # Step 2: List all files in that folder
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    if not file_list:
        print("No files found in the folder.")
        return

    os.makedirs(local_folder, exist_ok=True)

    # Step 3: Download each file
    expected_files = {
    "sentinel2_" + startDate + "_start.tif",
    "sentinel2_" + endDate + "_end.tif",
    "sentinel1_" + startDate + "_start.tif",
    "sentinel1_" + endDate + "_end.tif"
    }

    for file in file_list:
        filename = file['title']
        if filename in expected_files:
            filepath = os.path.join(local_folder, filename)
            print(f"Downloading: {filename} → {filepath}")
            file.GetContentFile(filepath)

    print("All files downloaded successfully.")

def clear_drive_folder(drive, folder_name='my-app-images'):
    # Find folder ID
    folder_list = drive.ListFile({
        'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not folder_list:
        print("Folder not found on Google Drive.")
        return

    folder_id = folder_list[0]['id']

    # List and delete all files in the folder
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        print(f"Deleting {file['title']} from Google Drive...")
        file.Delete()

def clear_local_folder(local_path='images', output_path='output_images'):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

def generate_pngs(file_path):
    def save_image(data, output_path, cmap='viridis', normalize=True):
        """Function to save an image from the model output."""
        # Normalize if requested
        if normalize:
            data = np.clip(data, 0, 100)  # Ensure it's within the range [0, 100]
            norm = Normalize(vmin=0, vmax=100)
        else:
            norm = None  # No normalization

        plt.imshow(data, cmap=cmap, norm=norm)
        plt.axis('off')  # Remove axes for better visualization
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    try:
        # Open the raster file
        with rasterio.open(file_path) as src:
            img = src.read()  # shape: (bands, height, width)

        # Extract maps
        change_map = img[0]
        semantic_t1 = img[1]
        semantic_t2 = img[2] if img.shape[0] > 2 else None  # Handle if missing

        # Save change detection output
        save_image(change_map, "static/change_map.png", cmap='hot')

        # Save semantic segmentation for T1
        save_image(semantic_t1, "static/semantic_t1.png", cmap='viridis')

        # Save semantic segmentation for T2 (only if T2 exists)
        if semantic_t2 is not None:
            save_image(semantic_t2, "static/semantic_t2.png", cmap='viridis')

        print("PNG Images saved successfully.")

    except Exception as e:
        print(f"Error generating PNGs: {e}")

def geotiff_to_png(tiff_path, output_dir="static/images"):
    if 'sentinel2_' not in tiff_path:
        return
    
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        blue, green, red = src.read(1), src.read(2), src.read(3)

        def normalize(array):
            return (array - array.min()) / (array.max() - array.min())

        rgb = np.dstack((normalize(red), normalize(green), normalize(blue)))
        if '_start' in tiff_path:
            png_name = 'start.png'
        else:
            png_name = 'end.png'

        png_path = os.path.join(output_dir, png_name)

        plt.imsave(png_path, rgb)
        print("saved: " + png_name + " successfully")
        return png_path

def generate_change_detection_map(s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path, model_path, output_folder):
    """Generates the change detection output image and saves it to a folder."""
    output_filename = "change_detection_output.tif"
    output_path = os.path.join(output_folder, output_filename)

    run_change_detection(s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path, output_path, model_path)
    print("Change detection processing complete.")
    return output_path

def calculate_ndbi_difference(start_image, end_image):
    def calculate_ndbi(image):
        return image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    ndbi_start = calculate_ndbi(start_image)
    ndbi_end = calculate_ndbi(end_image)
    ndbi_diff = ndbi_end.select('NDBI').subtract(ndbi_start.select('NDBI'))

    # Thresholding
    urban_increase = ndbi_diff.gt(0.15).selfMask() # Adjust threshold as needed.
    urban_decrease = ndbi_diff.lt(-0.15).selfMask()

    ndbi_vis = urban_increase.visualize(palette='red').blend(urban_decrease.visualize(palette='blue'))

    return ndbi_vis.getMapId()['tile_fetcher'].url_format

def download_sentinel_images(roi, startDate, endDate):
    """Downloads Sentinel-2 and Sentinel-1 mosaicked images with preprocessing."""

    S2_HARMONIZED = "COPERNICUS/S2_HARMONIZED"
    S2_CLOUDS = "COPERNICUS/S2_CLOUD_PROBABILITY"
    S1 = "COPERNICUS/S1_GRD"
    S2_BANDS = ['B2', 'B3', 'B4', 'B8']
    S1_BANDS = ['VV', 'VH']
    global status_message
    
    # def get_s2_image(from_date):
    #     to_date = ee.Date(from_date).advance(1, 'month')

    #     s2 = ee.ImageCollection(S2_HARMONIZED) \
    #         .filterDate(from_date, to_date) \
    #         .filterBounds(roi)

    #     clouds = ee.ImageCollection(S2_CLOUDS) \
    #         .filterDate(from_date, to_date) \
    #         .filterBounds(roi)

    #     join = ee.Join.saveFirst('cloud') \
    #         .apply(s2, clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

    #     def add_cloud(img):
    #         cloud = ee.Image(img.get('cloud'))
    #         cloud_score = cloud.select('probability')
    #         return ee.Image(img).addBands(cloud_score).set('cloudScore', cloud_score.reduceRegion(
    #             reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e12).get('probability'))

    #     processed = ee.ImageCollection(join).map(add_cloud)
    #     mosaic = processed.sort('cloudScore').mosaic().select(S2_BANDS) \
    #         .unitScale(0, 10000).clamp(0, 1).unmask().float()

    #     return mosaic

    def get_s2_image(from_date):
        to_date = ee.Date(from_date).advance(1, 'month')

        startImage = ee.ImageCollection(S2_HARMONIZED) \
            .filterDate(from_date, to_date) \
            .filterBounds(roi) \
            .sort('CLOUD_COVERAGE_ASSESSMENT') \
            .first() \
            .select(S2_BANDS) \
            .unitScale(0, 10000) \
            .clamp(0, 1) \
            .unmask() \
            .float()

        return startImage

    def get_s1_image(from_date):
        to_date = ee.Date(from_date).advance(1, 'month')
        s1 = ee.ImageCollection(S1) \
            .filterBounds(roi) \
            .filterDate(from_date, to_date) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

        asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

        s1 = ee.ImageCollection(ee.Algorithms.If(asc.size().gt(desc.size()), asc, desc))

        def mask_noise(img):
            return img.updateMask(img.gte(-25))

        s1 = s1.map(mask_noise)

        orbit_numbers = s1.aggregate_array('relativeOrbitNumber_start').distinct().getInfo()
        orbit_means = ee.ImageCollection([s1.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit)).mean()
                                          for orbit in orbit_numbers])

        mosaic = orbit_means.mosaic().select(S1_BANDS) \
            .unitScale(-25, 0).clamp(0, 1).unmask().float()

        return mosaic

    s2_start = get_s2_image(startDate)
    s2_end = get_s2_image(endDate)
    s1_start = get_s1_image(startDate)
    s1_end = get_s1_image(endDate)

    filenames = {
        "s2_start": f"sentinel2_{startDate}_start",
        "s2_end": f"sentinel2_{endDate}_end",
        "s1_start": f"sentinel1_{startDate}_start",
        "s1_end": f"sentinel1_{endDate}_end",
    }

    export_params = [
        (s2_start, filenames["s2_start"]),
        (s2_end, filenames["s2_end"]),
        (s1_start, filenames["s1_start"]),
        (s1_end, filenames["s1_end"]),
    ]

    drive = authenticate_drive()
    clear_local_folder()
    download_dir = "images"
    os.makedirs(download_dir, exist_ok=True)
    export_tasks = []
    clear_drive_folder(drive)

    for image, name in export_params:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name,
            scale=10,
            region=roi,
            fileFormat='GeoTIFF',
            folder='my-app-images',
            fileNamePrefix=name,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        export_tasks.append(task)
    status_message = "Generating images..."
    wait_for_tasks(export_tasks)
    status_message = "Downloading images..."
    download_files_from_drive(startDate, endDate, drive, folder_name='my-app-images', local_folder=download_dir)
    tiff_paths = [os.path.join(download_dir, f"{name}.tif") for name in filenames.values()]
    png_paths = [geotiff_to_png(tiff) for tiff in tiff_paths]
    print("png paths:")
    print(png_paths)

    return tiff_paths

def generate_map(roi_coords, startDate, endDate):
    try:
        roi = ee.Geometry.Polygon(roi_coords)
        sentinel = "COPERNICUS/S2_SR_HARMONIZED"
        landsat = "LANDSAT/LC09/C02/T1_L2"
        landsatBands = ['SR_B4', 'SR_B3', 'SR_B2']
        sentinelBands = ['B4', 'B3', 'B2']

        startImage = ee.ImageCollection(sentinel) \
            .filterDate(startDate, ee.Date(startDate).advance(1, 'month')) \
            .filterBounds(roi) \
            .sort('CLOUD_COVERAGE_ASSESSMENT')
        
        if startImage.size().getInfo() == 0:
            return jsonify({"error": "No images found for the given start time."}), 400

        endImage = ee.ImageCollection(sentinel) \
            .filterDate(endDate, ee.Date(endDate).advance(1, 'month')) \
            .filterBounds(roi) \
            .sort('CLOUD_COVERAGE_ASSESSMENT')
         
        if endImage.size().getInfo() == 0:
            return jsonify({"error": "No images found for the given end time."}), 400
        
        startImage = startImage.first()
        endImage = endImage.first()
        # 3000 for Sentinel, 20000 for LandSat - max range
        true_color_params = {
            'bands': sentinelBands,
            'min': 0,
            'max': 3000
        }

        startImageClipped = startImage.clip(roi)
        endImageClipped = endImage.clip(roi)

        start_image_url = startImageClipped.getMapId(true_color_params)['tile_fetcher'].url_format
        end_image_url = endImageClipped.getMapId(true_color_params)['tile_fetcher'].url_format

        ndbi_diff_url = calculate_ndbi_difference(startImageClipped, endImageClipped)

        center = startImageClipped.geometry().centroid().coordinates().getInfo()
        center_lat, center_lon = center[1], center[0]

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        folium.TileLayer(
            tiles=start_image_url,
            attr='Google Earth Engine',
            name='Start Date',
            overlay=True,
            control=True,
        ).add_to(m)

        folium.TileLayer(
            tiles=end_image_url,
            attr='Google Earth Engine',
            name='End Date',
            overlay=True,
            control=True,
        ).add_to(m)

        folium.TileLayer(
            tiles=ndbi_diff_url,
            attr='Google Earth Engine',
            name='Image Difference',
            overlay=True,
            control=True,
            show=False
        ).add_to(m)

        folium.LayerControl().add_to(m)
        return m._repr_html_()

    except Exception as e:
        return jsonify({"error": f"Error generating map: {e}"}), 400

def generate_long_term_map(city_name, startYear, endYear):
    try:
        cities = ee.FeatureCollection("TIGER/2018/Places")
        city = cities.filter(ee.Filter.eq('NAME', city_name)).first()
        if city is None or city.getInfo() is None:
            return jsonify({"error": f"City '{city_name}' not found."}), 400

        roi = city.geometry()

        startYear = int(startYear)
        endYear = int(endYear)

        def get_landsat_collection(year):
            if year < 2012:
                return "LANDSAT/LT05/C01/T1_SR"
            elif year < 2013:
                return "LANDSAT/LE07/C01/T1_SR"
            else:
                return "LANDSAT/LC08/C01/T1_SR"

        def get_ndvi(img):
            return img.normalizedDifference(['B5', 'B4']).rename('NDVI')

        def get_ndbi(img):
            return img.normalizedDifference(['B6', 'B5']).rename('NDBI')

        def get_composite(collection_id, year):
            return ee.ImageCollection(collection_id) \
                .filterBounds(roi) \
                .filterDate(f"{year}-01-01", f"{year}-12-31") \
                .sort('CLOUD_COVER') \
                .median() \
                .clip(roi)

        start_coll = get_landsat_collection(startYear)
        end_coll = get_landsat_collection(endYear)

        start_img = get_composite(start_coll, startYear)
        end_img = get_composite(end_coll, endYear)

        ndvi_diff = get_ndvi(end_img).subtract(get_ndvi(start_img)).rename('NDVI_Change')
        ndbi_diff = get_ndbi(end_img).subtract(get_ndbi(start_img)).rename('NDBI_Change')

        vis_rgb = {"bands": ['B4', 'B3', 'B2'], "min": 0, "max": 3000}
        vis_diff = {"min": -0.5, "max": 0.5, "palette": ['red', 'white', 'green']}

        center = roi.centroid().coordinates().getInfo()
        lat, lon = center[1], center[0]

        m = folium.Map(location=[lat, lon], zoom_start=11)

        folium.TileLayer(
            tiles=start_img.visualize(**vis_rgb).getMapId()['tile_fetcher'].url_format,
            attr='Start Year',
            name=f'{startYear} Image'
        ).add_to(m)

        folium.TileLayer(
            tiles=end_img.visualize(**vis_rgb).getMapId()['tile_fetcher'].url_format,
            attr='End Year',
            name=f'{endYear} Image'
        ).add_to(m)

        folium.TileLayer(
            tiles=ndvi_diff.visualize(**vis_diff).getMapId()['tile_fetcher'].url_format,
            attr='NDVI Change',
            name='NDVI Change'
        ).add_to(m)

        folium.TileLayer(
            tiles=ndbi_diff.visualize(**vis_diff).getMapId()['tile_fetcher'].url_format,
            attr='NDBI Change',
            name='NDBI Change'
        ).add_to(m)

        folium.LayerControl().add_to(m)
        return m._repr_html_()

    except Exception as e:
        return jsonify({"error": f"Error generating long-term map: {str(e)}"}), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_roi', methods=['POST'])
def process_roi():
    global status_message
    try:
        data = request.json
        roi_coords = data.get("roi")
        startDate = data.get("startDate")
        endDate = data.get("endDate")

        if not roi_coords:
            return jsonify({"error": "No ROI provided"}), 400

        roi = ee.Geometry.Polygon([roi_coords])
        model_path = "./"
        status_message = "Searching for satellite images..."
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path = download_sentinel_images(roi, startDate, endDate)
        
        output_folder = "output_images"
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = None
        try:
            status_message = "Generating change detection map..."
            output_image_path = generate_change_detection_map(
                s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path, model_path, output_folder
            )
            print(f"Output image saved to: {output_image_path}")
            status_message = "Generating images..."
            generate_pngs(output_image_path)
            status_message = "Image generation done."
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        status_message = "Generating map..."
        folium_map = generate_map(roi_coords, startDate, endDate)
        status_message = "Done"
        if folium_map:
            return jsonify({
                "message": "Sentinel map generated successfully",
                "folium_map": folium_map,
                "change_map": "Change detection image generated successfully",
                "output_image_path": output_image_path,
                "change_image_url": "/static/change_map.png",
                "start": "/static/images/start.png",
                "end": "/static/images/end.png"
            })
        else:
            return jsonify({"error": "Failed to generate Sentinel map."}), 500

    except (ValueError, SyntaxError) as e:
        return jsonify({"error": f"Invalid coordinate format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_long_term', methods=['POST'])
def long_term_change_detection():
    global status_message
    try:
        data = request.json
        roi_coords = data.get("coords")
        startYear = int(data.get("startYear"))
        endYear = int(data.get("endYear"))

        if not roi_coords:
            return jsonify({"error": "No ROI provided"}), 400

        roi = ee.Geometry.Polygon([roi_coords])
        
        center = roi.centroid().coordinates().getInfo()
        center_lat, center_lon = center[1], center[0]
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        status_message = "Generating satellite images..."
        for year in range(startYear, endYear + 1, 5):
            img = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").toBands().select(f"{year}_population_count")
            if(year == startYear): display = True
            else: display = False
            add_population_layer(img, f"Population {year}", roi, m, display)
        
        folium.LayerControl().add_to(m)
        map_html = m._repr_html_()
        status_message = "Calculating population count..."
        df = calculate_population(roi, startYear, endYear)
        status_message = "Done"
        return jsonify({"folium_map": map_html, "population_data": df})
    except (ValueError, SyntaxError) as e:
        return jsonify({"error": f"Invalid coordinate format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def get_status():
    return jsonify({"status": status_message})

if __name__ == '__main__':
    app.run(debug=True)