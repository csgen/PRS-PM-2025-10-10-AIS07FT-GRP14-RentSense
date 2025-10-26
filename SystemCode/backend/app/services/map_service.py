import folium
from app.models import PropertyLocation


async def fetch_map_page(*, location: PropertyLocation) -> str:
    
    coords = [location.latitude, location.longitude]
    zoom_level = 16

    m = folium.Map(
        location=coords,
        zoom_start=zoom_level,
        height="100%"
    )

    folium.Marker(
        location=coords,
        tooltip="Property Location"
    ).add_to(m)

    html_string = m.get_root().render()
    
    return html_string