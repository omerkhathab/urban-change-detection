let startLayer, endLayer;
// Initialize the map
var map = L.map('map').setView([13.04, 80.2], 13);

// ESRI satellite basemap layer (this is the background)
var esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; <a href="https://www.esri.com/en-us/arcgis/products/arcgis-online">ESRI</a>'
}).addTo(map);

// OpenStreetMap layer
var osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
});

// layer control to toggle between the satellite and OpenStreetMap layers
var baseMaps = {
    "Satellite": esriLayer,
    "OpenStreetMap": osmLayer
};

// Add the layer control to the map
L.control.layers(baseMaps).addTo(map);

// Feature group to store drawn items
var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

// Add draw control
var drawControl = new L.Control.Draw({
    draw: {
        polygon: false,
        polyline: false,
        rectangle: true,
        circle: false,
        marker: false
    },
    edit: {
        featureGroup: drawnItems
    }
});
map.addControl(drawControl);

// Capture drawn polygon
var selectedROI = null;
map.on('draw:created', function (event) {
    var layer = event.layer;
    drawnItems.clearLayers();  // Remove previous polygon
    drawnItems.addLayer(layer);
    selectedROI = layer.getLatLngs()[0];  // Get polygon coordinates
    console.log("ROI Coordinates:", selectedROI);
});

// Submit ROI coordinates to Flask
function submitROI() {
    if (!selectedROI) {
        alert("Please draw a polygon first!");
        return;
    }

    var coords = selectedROI.map(coord => [coord.lng, coord.lat]); // Convert to [lng, lat]
    var city = "no name";
    var startDate = document.getElementById("startDate").value;
    var endDate = document.getElementById("endDate").value;

    if (!startDate || !endDate) {
        document.getElementById("result").innerText = "Please select both start and end dates.";
        return;
    }
    if (new Date(startDate) > new Date(endDate)) {
        document.getElementById("result").innerText = "Start date must be before end date.";
        return;
    }
    document.getElementById("result").innerText = `Start: ${startDate}\nEnd: ${endDate}`;
    document.getElementById("response").innerText = "Search for satellite images...";
    fetch('/process_roi', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi: coords, city, startDate, endDate})
    })
    .then(response => response.json())
    .then(data => {
        console.log("Server Response:", data);
        if (data.folium_map) {
            document.getElementById('folium-map-container').innerHTML = data.folium_map;
            document.getElementById("response").innerText = "";
        } else {
            document.getElementById("response").innerText = "Error: " + JSON.stringify(data);
        }
        if (data.change_image_url) {
            document.getElementById("result-images").innerHTML = `
                <img src="${data.change_image_url}" alt="Change Detection">
                <img src="${data.start}" alt="start image">
                <img src="${data.end}" alt="end image">
            `;
        }

    })
    .catch(error => console.error("Error:", error));
}