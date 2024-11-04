import osmnx as ox
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from shapely.geometry import Point

import numpy as np
from shapely.geometry import box
from scipy.spatial import cKDTree

shapefile_path = 'inputs/Syracuse_Neighborhoods.shp'
quartiers = gpd.read_file(shapefile_path)

if quartiers.crs != 'EPSG:4326':
    quartiers = quartiers.to_crs(epsg=4326)

amenities = ['hospital', 'clinic', 'doctors']

ville = ox.geocode_to_gdf('Syracuse, New York, USA')
ville = ville.to_crs(epsg=4326)
limites_ville = ville.geometry.unary_union

tags = {'amenity': amenities}
pois = ox.features_from_polygon(limites_ville, tags)

pois_sante = pois[pois['amenity'].isin(amenities)]

quartiers['score_sante'] = 0

hospi_sante = pois_sante[pois_sante['amenity'].isin(['hospital', 'clinic'])]

doctors_sante = pois_sante[pois_sante['amenity'] == 'doctors']


hospi_sante = hospi_sante.copy()
hospi_sante['geometry'] = hospi_sante.geometry.apply(lambda geom: geom.centroid if geom.geom_type != 'Point' else geom)

hospi_coords = np.array([[point.x, point.y] for point in hospi_sante.geometry])
hospi_tree = cKDTree(hospi_coords)

for idx, quartier in quartiers.iterrows():
    minx, miny, maxx, maxy = quartier.geometry.bounds

    x_grid = np.arange(minx, maxx, 0.001) 
    y_grid = np.arange(miny, maxy, 0.001) 
    grid_cells = [box(x, y, x + 0.001, y + 0.001) for x in x_grid for y in y_grid]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=quartiers.crs)

    score_influence = 0

    for cell in grid.geometry:
        hop_in_cell = hospi_sante[hospi_sante.within(cell)]
        if not hop_in_cell.empty:
            score_influence += 1
        else:
            centroid = cell.centroid
            distance, _ = hospi_tree.query([centroid.x, centroid.y])
            if distance > 0:
                score_influence += 1 / distance

    number_of_doctors = doctors_sante[doctors_sante.within(quartier.geometry)].shape[0]

    score_influence *= np.log(max(number_of_doctors, 3))

    quartiers.at[idx, 'score_influence'] = score_influence

quartiers['score'] = (quartiers['score_influence'] - quartiers['score_influence'].min()) / (quartiers['score_influence'].max() - quartiers['score_influence'].min())

quartiers.to_file('output/neighboors_health.shp', driver='ESRI Shapefile')