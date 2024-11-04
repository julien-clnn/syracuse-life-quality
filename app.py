# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_folium import st_folium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from shapely.geometry import Point
from shapely.wkt import loads
import networkx as nx
import osmnx as ox
import contextily as ctx
import folium

from pyngrok import ngrok

def sidebar():
    st.sidebar.header('Map Options')

    map_option = st.sidebar.selectbox(
        "Choose the map to display:",
        ("Global Score", "Athletics Score", "Safety Score", "Health Access Score")
    )

    with st.sidebar.form("my_form"):
        submitted = st.form_submit_button("Calculate")

    return submitted, map_option

def display_map_all_neighborhoods(dataset_with_score, score_column):
    m = folium.Map(location=[43.0481, -76.1474], zoom_start=12, tiles="OpenStreetMap")

    folium.Choropleth(
        geo_data=dataset_with_score,
        name="choropleth",
        data=dataset_with_score,
        columns=["Name", score_column],
        key_on="feature.properties.Name",
        fill_color="RdYlGn",
        fill_opacity=0.8,
        line_weight=0 
    ).add_to(m)

    folium.GeoJson(
        dataset_with_score,
        name="Neighborhoods",
        tooltip=folium.GeoJsonTooltip(
            fields=["Name", score_column],
            aliases=["Neighborhood:", "Score:"],
            localize=True
        ),
        style_function=lambda feature: {
            'fillColor': feature['properties'].get('color', '#gray'),  
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.1,  
            'opacity': 0.3  
        }
    ).add_to(m)

    st_folium(m, width=700, height=500)

def load_health_data():
    """
    Load and project the health data shapefile.
    """
    filepath = "data/neighboors_health.shp"
    health_data = gpd.read_file(filepath)
    health_data = health_data.to_crs(epsg=4326)
    return health_data

def load_scores_data():
    """
    Load and project the athletics and crime score data shapefile.
    """
    filepath = "data/df_with_scores.shp"
    scores_data = gpd.read_file(filepath)
    scores_data = scores_data.to_crs(epsg=4326)
    scores_data['score_crim'] = 1 - scores_data['score_crim']
    return scores_data

def merge_health_scores(scores_df, health_df):
    """
    Merge health data with athletics and crime scores on the 'Name' column.
    Rename the health score column to 'score_health'.
    """
    merged_df = scores_df.merge(health_df[['Name', 'score']], on='Name', how='left')
    merged_df = merged_df.rename(columns={'score': 'score_health'})
    return merged_df

def min_max_normalize(column):
    """
    Normalize a series to a 0-1 range using min-max normalization.
    """
    return (column - column.min()) / (column.max() - column.min())

def calculate_area_weighted_score(df):
    """
    Calculate the area of each neighborhood and use it to weight the scores,
    reducing the impact of area by half.
    """
    df['score'] = (df['score_athl'] + df['score_crim'] + df['score_health'])/3

    return df

def main(dataset_with_score):
    st.title('Quality of Life Score at Syracuse 🇺🇸')
    st.subheader('by Julien Calonne 🇫🇷 & Zerin Arif 🇳🇱')

    st.markdown(
        """
        Welcome to our quality of life map!
        <br><br/>
        We have designed this map to help you understand the quality of life in your area of Syracuse.

        Explore the predictions using the filters on the left.

        To read more about how the model works, click below.
        """,
        unsafe_allow_html=True
    )

    if "show_details" not in st.session_state:
        st.session_state["show_details"] = False

    if st.session_state["show_details"]:
        st.sidebar.button("Back", on_click=lambda: st.session_state.update({"show_details": False}))

        st.markdown("""
        ### Method for Calculating Quality of Life per Neighborhood

        Our approach to assessing quality of life across neighborhoods in Syracuse involves examining key attributes like crime, athletic parks, and safety. We calculate each attribute’s density within each neighborhood to create a proportional measure, making neighborhoods comparable regardless of size.

        1. ⁠**Attribute Density Calculation:**
        For attributes such as crime and athletic parks, we calculate *attribute density* to represent each neighborhood’s contribution to Syracuse’s total occurrences of that attribute. This density provides a proportional measure for each neighborhood on a relative scale.
        """)

        st.latex(r"\text{Attribute Density } = \frac{\text{Number of occurances in neighborhood}}{\text{Total number of occurances in all neighborhoods}}")

        st.markdown("""
        This proportion reveals how much each neighborhood contributes to the citywide count for a specific attribute, allowing us to compare neighborhoods in relative terms.

        2. ⁠**Accounting for Border Effects:**
        To address attributes on neighborhood borders, which might otherwise only count for one neighborhood, we apply a buffer to extend the impact area. We use a buffer of 50 meters for crime and 500 meters for parks, ensuring that occurrences near boundaries are fairly represented across neighboring areas.
        """)

        st.markdown("""
        3. **Adding Healthcare accessibility:**
        The health score for each neighborhood is calculated by analyzing the presence and proximity of hospitals and other health facilities within a grid overlaying the neighborhood. Scores are influenced by the number of facilities found in each grid cell and the distance to the nearest facility.
        """)

        st.markdown("""
        4. ⁠**Get weighted average of all attributes:**
        After we have the score of all the attributes separately we calculate the average to get the total score of a neighborhood
        """)

        st.markdown("""
        ### Points for Improvements

        - **Crime Weighting:** To increase precision, we add a weight to crime occurrences. More severe crimes, such as homicide, are given a higher weight compared to less severe crimes, like fraud.

        - ⁠**Street Segment Analysis with OSMnx:** Initially, our plan was to analyze quality of life metrics on a street segment level using the OSMnx library. However, due to high computation time and limited resources, we decided to focus on neighborhoods instead.
        ![POC](https://raw.githubusercontent.com/julien-clnn/syracuse_open_data/main/streetsegment_example.png "Proof of concept using 2 Datasets and buffers").

        - ⁠**Weighted Averages for Attributes:** For the weighted average, we thought of three options:
            - *Normal Average*: A simple average of all attribute scores.
            - *Custom Weighted Average*: Allows users to prioritize attributes based on personal or policy-based preferences.
            - *Area-Based Weighting*: Adjusts the scores based on neighborhood size, so larger neighborhoods aren’t overly penalized for higher counts of certain attributes like crime.

            For now, we use the normal average method due to limited domain knowledge on ideal weighting schemes.

        - ⁠**Add More Datasets:** Expanding the number of attributes included in the quality of life calculation.
        - **User-Defined Buffers:** Allow users to set custom buffer distances for crime or park occurrences.
        - **Distance Decay:** Implement a system where the influence of an occurrence (e.g., a park) decreases with distance. For instance, streets close to a park would receive a higher score, while those farther away would receive a lower score.
        """)

        st.markdown("""
        ### Data Used

        - ⁠Parks:
          [Parks Dataset](https://data.syr.gov/datasets/077ba4795c6a4229a3f51189f8113079_0/explore?location=43.017372%2C-76.088370%2C11.77)

        - ⁠Crime:
          [Crime Dataset 1](https://data.syr.gov/datasets/94bc33f1a2c646b995d1ab356edf0730_0/about),
          [Crime Dataset 2](https://data.syr.gov/datasets/9765b20578304b95b118e3e3f6db4d0e_0/about)

        - ⁠Healthcare:
          [Osmnx Points of Interest](https://stackoverflow.com/questions/61639039/using-osmnx-to-retrieve-nearby-points-of-interest)
        """)

    else:
        if st.button("More details"):
            st.session_state["show_details"] = True

        submitted, map_option = sidebar()

        if map_option == "Athletics Score":
            score_column = 'score_athl'
        elif map_option == "Safety Score":
            score_column = 'score_crim'
        elif map_option == "Health Access Score":
            score_column = 'score_health'
        else:
            score_column = 'score'

        if submitted or st.session_state.get("map_shown", False):
            display_map_all_neighborhoods(dataset_with_score, score_column)
            st.session_state["map_shown"] = True

health_neigh_with_score = load_health_data()

merged_df = load_scores_data()

merged_df = merge_health_scores(merged_df, health_neigh_with_score)

merged_df = calculate_area_weighted_score(merged_df)

if __name__ == "__main__":
    main(merged_df)
