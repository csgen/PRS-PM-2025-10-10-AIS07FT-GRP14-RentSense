// Fallback map HTML for Singapore Central Area
// Used when backend API fails to provide map data
export const FALLBACK_MAP_HTML = `
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
    }
    #map {
      position: absolute;
      top: 0;
      bottom: 0;
      right: 0;
      left: 0;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    var map = L.map('map').setView([1.3521, 103.8198], 12);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    }).addTo(map);
    
    var marker = L.marker([1.3521, 103.8198]).addTo(map);
    marker.bindPopup("<b>Singapore Central Area</b><br>").openPopup();
  </script>
</body>
</html>
`
