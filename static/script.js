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
        document.getElementById("response").innerText = "Please select both start and end dates.";
        return;
    }
    if (new Date(startDate) > new Date(endDate)) {
        document.getElementById("response").innerText = "Start date must be before end date.";
        return;
    }
    fetchStatus();
    fetch('/process_roi', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi: coords, city, startDate, endDate})
    })
    .then(response => response.json())
    .then(data => {
        console.log("Server Response:", data);
        if (data.folium_map) {
            mapContainer.classList.remove("d-none");
            mapContainer.innerHTML = data.folium_map;
            document.getElementById("response").innerText = "";
        } else {
            document.getElementById("response").innerText = "There was an error: " + data.error;
        }
        if (data.change_image_url) {
            document.getElementById("result-images").innerHTML = `
                <div class="d-flex flex-row w-100 gap-3">
                    <div id="sliderContainer" class="flex-fill"></div>
                    <div class="flex-fill d-flex align-items-center justify-content-center">
                        <img src="${data.change_image_url}" alt="Change Map" class="img-fluid shadow rounded" style="max-height: 100%; max-width: 100%; object-fit: contain;">
                    </div>
                </div>
            `;
        
            new juxtapose.JXSlider('#sliderContainer', [
                {
                    src: data.start,
                    label: 'Before'
                },
                {
                    src: data.end,
                    label: 'After'
                }
            ], {
                animate: true,
                showLabels: true,
                showCredits: false,
                makeResponsive: true
            });
        }
    })
    .catch(error => console.error("Error:", error));
}

function submitLongTerm() {
    if (!selectedROI) {
        alert("Please draw a polygon first!");
        return;
    }
    var coords = selectedROI.map(coord => [coord.lng, coord.lat]);
    var startYear = document.getElementById("startYear").value;
    var endYear = document.getElementById("endYear").value;
    if (startYear > endYear) {
        document.getElementById("response").innerText = "Start date must be before end date.";
        return;
    }
    fetch('/process_long_term', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coords, startYear, endYear})
    })
    .then(response => response.json())
    .then(data => {
        if (data.folium_map) {
            const mapContainer = document.getElementById("folium-map-container");
            mapContainer.classList.remove("d-none");
            mapContainer.innerHTML = data.folium_map;

            const popDiv = document.createElement("div");
            popDiv.className = "p-3 my-2";
            popDiv.innerHTML = "<h5 class='mb-1'>Population Over Time (Estimate)</h5><table class='table table-bordered'><thead><tr><th>Date</th><th>Population</th></tr></thead><tbody></tbody></table>";
            const tbody = popDiv.querySelector("tbody");

            data.population_data.forEach(entry => {
                const row = `<tr><td>${entry.date}</td><td>${parseInt(entry.pop).toLocaleString()}</td></tr>`;
                tbody.innerHTML += row;
            });

            mapContainer.appendChild(popDiv);
            document.getElementById("response").innerText = "";
        } else {
            document.getElementById("response").innerText = "There was an error: " + data.error;
        }
    })
}
function fetchStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('status-box').innerText = data.status;
            if (data.status !== "Done") {
                document.getElementById('loading').classList.remove("d-none");
                document.getElementById('loading').classList.add("d-flex");
                setTimeout(fetchStatus, 3000);
            } else {
                document.getElementById('loading').classList.remove("d-flex");
                document.getElementById('loading').classList.add("d-none");
            }
    });
}