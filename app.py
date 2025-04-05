from flask import Flask, render_template, request, jsonify, send_from_directory
import ee
import geemap
import os
import folium
import ast

ee.Authenticate()
ee.Initialize(project="urban-change-detection")

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

def create_map():
    m = geemap.Map(center=[13.04, 80.2], zoom=13)
    # m = geemap.Map(zoom=13)
    m.add_basemap("SATELLITE")
    m.to_html("static/map.html")

def calculate_image_difference(start_image, end_image, bands, stdDevThreshold=2.0, smoothingRadius=3, minObjectSize=10):
    """
    Calculates the difference between two images, applies a statistical threshold,
    and visualizes significant changes as a heatmap.
    """
    try:
        print(start_image.geometry().getInfo())
        # Calculate the difference for the desired bands.
        diff_image = end_image.select(bands).subtract(start_image.select(bands))

        # Calculate the magnitude of the changes.
        magnitude = diff_image.abs().reduce(ee.Reducer.sum())

        # Smooth the magnitude image to reduce noise.
        smoothedMagnitude = magnitude.convolve(ee.Kernel.circle(smoothingRadius))

        # Calculate the mean and standard deviation.
        mean = ee.Number(smoothedMagnitude.reduceRegion(ee.Reducer.mean(), geometry=start_image.geometry(), scale=30).get('mean')) #Corrected key
        stdDev = ee.Number(smoothedMagnitude.reduceRegion(ee.Reducer.stdDev(), geometry=start_image.geometry(), scale=30).get('stdDev'))

        # Calculate the threshold based on standard deviations.
        threshold = mean.add(stdDev.multiply(stdDevThreshold))

        # Apply a threshold to create a mask.
        mask = smoothedMagnitude.gt(threshold)

        # Remove small, isolated changes.
        connectedPixels = mask.connectedPixelCount(minObjectSize)
        filteredMask = mask.updateMask(connectedPixels.gte(minObjectSize))

        # Mask out areas with minimal change.
        masked_diff = smoothedMagnitude.updateMask(filteredMask)

        maxValue = ee.Number(magnitude.reduceRegion(ee.Reducer.max(), geometry=start_image.geometry(), scale=30).get('sum'))

        # Visualize the masked difference as a heatmap.
        heatmap_vis = masked_diff.visualize(
            min=threshold,
            max=maxValue,
            palette=['blue', 'yellow', 'red']
        )

        # Get the map ID dictionary.
        map_id_dict = heatmap_vis.getMapId()

        # Construct the tile URL.
        tile_url = map_id_dict['tile_fetcher']['url_format']

        # Return both the map ID dictionary and the tile URL.
        return {
            'map_id': map_id_dict,
            'tile_url': tile_url
        }

    except Exception as e:
        print(f"Error in calculate_image_difference: {e}")
        return None

def calculate_band_ratio(start_image, end_image, bands):
    return "url 2"
    ratio_image = end_image.divide(start_image)
    ratio_vis_params = {
        'bands': bands,
        'min': 0.8,
        'max': 1.2,
        'palette': ['blue', 'white', 'red']
    }
    ratio_image_vis = ratio_image.visualize(ratio_vis_params)
    return ratio_image_vis.getMapId()['tile_fetcher'].url_format

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

"""
def generate_image(roi_coords, city):
    roi = ee.Geometry.Polygon(roi_coords)
    sentinel = "COPERNICUS/S2_SR_HARMONIZED"
    landsat = "LANDSAT/LC09/C02/T1_L2"
    landsatBands = ['SR_B4', 'SR_B3', 'SR_B2']
    sentinelBands = ['B4', 'B3', 'B2']

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # start = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    #     .filterBounds(roi) \
    #     .filterDate(startDate, ee.Date(startDate).advance(1, 'month')) \
    #     .sort('CLOUD_COVERAGE_ASSESSMENT') \
    #     .first()

    # end = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    #     .filterBounds(roi) \
    #     .filterDate(endDate, ee.Date(endDate).advance(1, 'month')) \
    #     .sort('CLOUD_COVERAGE_ASSESSMENT') \
    #     .first()

    filtered = collection.filterBounds(roi) \
                        .filterDate('2023-01-01', '2023-12-31') \
                        .sort('CLOUD_COVERAGE_ASSESSMENT') \
                        .first()

    print(f"Selected image ID: {filtered.get('system:id').getInfo()}")
    print(f"Image date: {ee.Date(filtered.get('system:time_start')).format('YYYY-MM-dd').getInfo()}")

    viz_params = {
        'bands': ['B4', 'B3', 'B2'],  # RGB bands
        'min': 0,
        'max': 3000,  # Adjust based on your image data range
        'gamma': 1.4
    }

    rgb_image = filtered.visualize(**viz_params).uint8()

    task = ee.batch.Export.image.toDrive(
        image=rgb_image,
        description=f'sentinel2_{city}',
        scale=10,
        region=roi,
        fileFormat='GeoTIFF'
    )
    task.start()
    return jsonify({
        "message": "ROI received successfully",
        "city": city,
        "roi_geometry": roi.getInfo()
    })
"""

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

        # masked_image = maskS2clouds(image)
        startImageClipped = startImage.clip(roi)
        endImageClipped = endImage.clip(roi)

        start_image_url = startImageClipped.getMapId(true_color_params)['tile_fetcher'].url_format
        end_image_url = endImageClipped.getMapId(true_color_params)['tile_fetcher'].url_format

        # diff_image_data = calculate_image_difference(startImageClipped, endImageClipped, sentinelBands, stdDevThreshold=2.0, smoothingRadius=3, minObjectSize=10)
        # if diff_image_data:
        #     diff_image_url = diff_image_data['tile_url']
        #     print(f"Difference image URL: {diff_image_url}")
        # else:
        #     print("Failed to get difference image URL.")
        #     return "Error generating map."
        # ratio_image_url = calculate_band_ratio(startImageClipped, endImageClipped, sentinelBands)
        ndbi_diff_url = calculate_ndbi_difference(startImageClipped, endImageClipped)

        # print(diff_image_url)
        # print("ratio url: " + ratio_image_url)
        # print("ndbi url: " + ndbi_diff_url)

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

        # folium.TileLayer(
        #     tiles=ratio_image_url,
        #     attr='Google Earth Engine',
        #     name='Band Ratio',
        #     overlay=True,
        #     control=True,
        # ).add_to(m)

        # folium.TileLayer(
        #     tiles=ndbi_diff_url,
        #     attr='Google Earth Engine',
        #     name='NDVI Difference',
        #     overlay=True,
        #     control=True,
        # ).add_to(m)

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

        folium_map = generate_map(roi_coords, startDate, endDate)

        if folium_map:
            return jsonify({
                "message": "Landsat map generated successfully",
                "folium_map": folium_map
            })
        else:
            return jsonify({"error": "Failed to generate Landsat map."}), 500

    except (ValueError, SyntaxError) as e:
        return jsonify({"error": f"Invalid coordinate format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)