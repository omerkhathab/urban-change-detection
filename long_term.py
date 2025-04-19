import ee
import folium
import pandas as pd

def add_population_layer(image, name, roi, fmap, display=True):
    population_map = image.clip(roi)
    map_url = population_map.getMapId()['tile_fetcher'].url_format
    folium.TileLayer(
        tiles=map_url,
        attr='GHSL Population',
        name=name,
        overlay=True,
        control=True,
        show=display
    ).add_to(fmap)

def calculate_population(roi, startYear, endYear):
    def pop_count(img):
        pop_sum = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=100).values().get(0)
        year = img.date().get('year')
        return ee.Feature(None, {'date': year.format(), 'pop': pop_sum})
    
    pop = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
    
    # Filter by year before mapping
    pop = pop.filter(ee.Filter.calendarRange(startYear, endYear, 'year'))
    
    pop_val = pop.map(pop_count)
    feature_list = pop_val.toList(pop_val.size()).getInfo()

    df = pd.DataFrame({
        'date': [f['properties']['date'] for f in feature_list],
        'pop': [f['properties']['pop'] for f in feature_list]
    })
    df = df.dropna()
    df['date'] = pd.to_numeric(df['date'], errors='coerce')  # Ensure it's numeric
    df = df.sort_values('date')
    df = df.round(0).reset_index(drop=True)
    
    return df.to_dict(orient='records')