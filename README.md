## Access the project : 
https://syracuse-quality-of-life.streamlit.app/

## About the Project
This project aims to calculate a quality of life score for neighborhoods in Syracuse, New York, using data on safety, parks, and healthcare accessibility. The tool offers insights by providing a score that can support urban planning or other initiatives.

## Inspiration
The inspiration came from the desire to quantify quality of life in a way that includes both positive and negative characteristics. We wanted to create a measure that can help identify areas of improvement for the city.

## How we built it
We built this tool using Python, Streamlit, and geospatial libraries like GeoPandas and OSMnx.

## Challenges we ran into
The biggest challenge was computational efficiency, especially when working with fine-grained street segment data using OSMnx. We also faced difficulties in determining appropriate weights for the attributes without domain-specific knowledge.

## Accomplishments that we're proud of
We successfully developed a user-friendly tool that highlights neighborhood characteristics in an accessible format. We’re proud of our ability to develop a working model with limited resources and within a short timeframe.

## What we learned
This project helped us improve our skills in data processing, geospatial analysis, and dashboard making. We also gained insight into the complexity of quantifying quality of life and the importance of data-driven decision-making in urban planning.

## What's next for Syracuse Quality of Life Score per Neighborhoods
Future steps include adding more datasets, refining attribute weights, allowing users to set custom buffers, and implementing a distance decay model to make this tool more adaptable and accurate in reflecting neighborhood quality of life.
