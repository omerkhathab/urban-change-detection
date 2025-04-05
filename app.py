from flask import Flask, render_template, request, jsonify, send_from_directory
import ee
import geemap
import os
import folium
from change_detection import run_change_detection

ee.Authenticate()
ee.Initialize(project="urban-change-detection")

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

def create_map():
    m = geemap.Map(center=[13.04, 80.2], zoom=13)
    # m = geemap.Map(zoom=13)
    m.add_basemap("SATELLITE")
    m.to_html("static/map.html")

def generate_change_detection_map(s2_t1_path, s2_t2_path, model_path, output_folder):
    """Generates the change detection output image and saves it to a folder."""
    output_filename = "change_detection_output.tif"
    output_path = os.path.join(output_folder, output_filename)

    # run_change_detection(s2_t1_path, s2_t2_path, output_path, model_path)
    print("Does this work?")
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
    
def calculate_ndbi_difference_refined(start_image, end_image):
    def calculate_ndbi(image):
        return image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    ndbi_start = calculate_ndbi(start_image)
    ndbi_end = calculate_ndbi(end_image)
    ndbi_diff = ndbi_end.select('NDBI').subtract(ndbi_start.select('NDBI'))

    # Adaptive Thresholding
    mean = ndbi_diff.reduceRegion(ee.Reducer.mean(), geometry=start_image.geometry(), scale=30).get('NDBI')
    stdDev = ndbi_diff.reduceRegion(ee.Reducer.stdDev(), geometry=start_image.geometry(), scale=30).get('NDBI')

    # dynamically change the threshold depending on the stdDev.
    positive_threshold = ee.Number(mean).add(ee.Number(stdDev).multiply(0.5))  # Adjust multiplier as needed
    negative_threshold = ee.Number(mean).subtract(ee.Number(stdDev).multiply(0.5))

    urban_increase = ndbi_diff.gt(positive_threshold).selfMask()
    urban_decrease = ndbi_diff.lt(negative_threshold).selfMask()

    # Morphological Operations (Noise Reduction)
    kernel = ee.Kernel.circle(1)  # Adjust radius as needed
    urban_increase_filtered = urban_increase.focal_mode(kernel).focal_max(kernel)
    urban_decrease_filtered = urban_decrease.focal_mode(kernel).focal_min(kernel)

    # Visualization Enhancement
    ndbi_vis = ee.ImageCollection([
        urban_increase_filtered.visualize(palette='red'),
        urban_decrease_filtered.visualize(palette='blue'),
        ndbi_diff.visualize(min=-0.2, max=0.2, palette=['blue', 'white', 'red']).unmask() #add the original ndbi diff as well.
    ]).mosaic()

    return ndbi_vis.getMapId()['tile_fetcher'].url_format

def download_sentinel_images(roi, startDate, endDate):
    """Downloads Sentinel-2 and Sentinel-1 images for the given ROI and date range."""

    # Sentinel-2 Images
    sentinel2 = "COPERNICUS/S2_SR_HARMONIZED"
    sentinel2Bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    startImageS2 = ee.ImageCollection(sentinel2) \
        .filterDate(startDate, ee.Date(startDate).advance(1, 'month')) \
        .filterBounds(roi) \
        .sort('CLOUD_COVERAGE_ASSESSMENT') \
        .first()

    endImageS2 = ee.ImageCollection(sentinel2) \
        .filterDate(endDate, ee.Date(endDate).advance(1, 'month')) \
        .filterBounds(roi) \
        .sort('CLOUD_COVERAGE_ASSESSMENT') \
        .first()

    # Sentinel-1 Images
    sentinel1 = "COPERNICUS/S1_GRD"
    sentinel1Bands = ['VH', 'VV']

    startImageS1 = ee.ImageCollection(sentinel1) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
        .filterDate(startDate, ee.Date(startDate).advance(1, 'month')) \
        .filterBounds(roi) \
        .sort('system:time_start') \
        .first()

    endImageS1 = ee.ImageCollection(sentinel1) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
        .filterDate(endDate, ee.Date(endDate).advance(1, 'month')) \
        .filterBounds(roi) \
        .sort('system:time_start') \
        .first()

    # Create download directory
    download_dir = "images"
    os.makedirs(download_dir, exist_ok=True)

    # File names
    s2_start_filename = f"sentinel2_{startDate}_start.tif"
    s2_end_filename = f"sentinel2_{endDate}_end.tif"
    s1_start_filename = f"sentinel1_{startDate}_start.tif"
    s1_end_filename = f"sentinel1_{endDate}_end.tif"

    # File paths
    s2_t1_path = os.path.join(download_dir, s2_start_filename)
    s2_t2_path = os.path.join(download_dir, s2_end_filename)
    s1_t1_path = os.path.join(download_dir, s1_start_filename)
    s1_t2_path = os.path.join(download_dir, s1_end_filename)

    # Export Sentinel-2 images
    task1_s2 = ee.batch.Export.image.toDrive(
        image=startImageS2.select(sentinel2Bands),
        description=s2_start_filename,
        scale=10,
        region=roi,
        fileFormat='GeoTIFF',
        folder='my-app-images',
        fileNamePrefix=s2_start_filename
    )
    task1_s2.start()

    task2_s2 = ee.batch.Export.image.toDrive(
        image=endImageS2.select(sentinel2Bands),
        description=s2_end_filename,
        scale=10,
        region=roi,
        fileFormat='GeoTIFF',
        folder='my-app-images',
        fileNamePrefix=s2_end_filename
    )
    task2_s2.start()

    # Export Sentinel-1 images
    task1_s1 = ee.batch.Export.image.toDrive(
        image=startImageS1.select(sentinel1Bands),
        description=s1_start_filename,
        scale=10,
        region=roi,
        fileFormat='GeoTIFF',
        folder='my-app-images',
        fileNamePrefix=s1_start_filename
    )
    task1_s1.start()

    task2_s1 = ee.batch.Export.image.toDrive(
        image=endImageS1.select(sentinel1Bands),
        description=s1_end_filename,
        scale=10,
        region=roi,
        fileFormat='GeoTIFF',
        folder='my-app-images',
        fileNamePrefix=s1_end_filename
    )
    task2_s1.start()

    print("download images from drive and save it into images folder.")

    return s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path

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
        ).add_to(m)

        folium.LayerControl().add_to(m)

        return m._repr_html_()

    except Exception as e:
        return jsonify({"error": f"Error generating map: {e}"}), 400

@app.route('/')
def index():
    """Render the main page with the interactive map."""
    create_map()
    return render_template('index.html')

@app.route('/map')
def get_map():
    """Serve the generated map HTML file."""
    return send_from_directory('static', 'map.html')

@app.route('/process_roi', methods=['POST'])
def process_roi():
    try:
        data = request.json
        roi_coords = data.get("roi")
        city = data.get("city")
        startDate = data.get("startDate")
        endDate = data.get("endDate")

        if not roi_coords:
            return jsonify({"error": "No ROI provided"}), 400

        roi = ee.Geometry.Polygon([roi_coords])
        model_path = "model.pt"
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path = download_sentinel_images(roi, startDate, endDate)
        
        output_folder = "output_images"
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = generate_change_detection_map(s2_t1_path, s2_t2_path, model_path, output_folder)

        folium_map = generate_map(roi_coords, startDate, endDate)

        if folium_map:
            return jsonify({
                "message": "Landsat map generated successfully",
                "folium_map": folium_map,
                "also": "Change detection image generated successfully",
                "output_image_path": output_image_path
            })
        else:
            return jsonify({"error": "Failed to generate Sentinel map."}), 500

    except (ValueError, SyntaxError) as e:
        return jsonify({"error": f"Invalid coordinate format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)