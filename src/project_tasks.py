from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from geopy.geocoders import Nominatim
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
from pyspark.sql.functions import lit
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import explode, col
import folium
import pandas as pd
import geopandas as gpd
import numpy as np
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from wordcloud import WordCloud

# Initialize Spark session
spark = SparkSession.builder \
    .appName("COVID-19 Twitter Analysis") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Load JSON data into DataFrame
tweet_df = spark.read.json("out.json")

tweet_df.show()

##############TASK 1

# Define a function for sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify sentiment based on polarity
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Define a UDF (User Defined Function) for sentiment analysis
sentiment_udf = udf(get_sentiment, StringType())

# Apply sentiment analysis UDF to the 'full_text' column
tweet_df = tweet_df.withColumn("sentiment", sentiment_udf(col("full_text")))

# Aggregate sentiment analysis results
sentiment_counts = tweet_df.groupBy("sentiment").count()

# Show aggregated results
sentiment_counts.show()

# Convert DataFrame to Pandas DataFrame for visualization
sentiment_counts_pd = sentiment_counts.toPandas()

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts_pd['sentiment'], sentiment_counts_pd['count'], color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis of COVID-19 Vaccine Discussions')
plt.xlabel('Sentiment')
plt.ylabel('Count')
# Save the plot as a PNG image
plt.savefig("sentiment_distribution_project.png")

########### TASK 2
# Function to filter tweets related to specific variants
def filter_variant(text, variant):
    # Case-insensitive matching for variant names
    if re.search(r'\b{}\b'.format(variant), text, re.IGNORECASE):
        return True
    else:
        return False

# Define variants
variants = ['Delta', 'Omicron']

# Define UDF for filtering tweets related to specific variants
filter_variant_udf = udf(lambda text, variant: filter_variant(text, variant), BooleanType())

# Perform sentiment analysis for each variant
for variant in variants:
    variant_tweet_df = tweet_df.filter(filter_variant_udf(col("full_text"), lit(variant)))
    variant_tweet_df = variant_tweet_df.withColumn("sentiment_" + variant.lower(), sentiment_udf(col("full_text")))

    # Aggregate sentiment analysis results for each variant
    sentiment_counts_variant = variant_tweet_df.groupBy("sentiment_" + variant.lower()).count()

    # Show aggregated results for each variant
    print("Sentiment Analysis for", variant, "Variant:")
    sentiment_counts_variant.show()

    # Visualization for each variant
    sentiment_counts_variant_pd = sentiment_counts_variant.toPandas()
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_counts_variant_pd['sentiment_' + variant.lower()], sentiment_counts_variant_pd['count'], color=['green', 'red', 'blue'])
    plt.title('Sentiment Analysis of ' + variant + ' Variant')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Save the figure
    plt.savefig(variant.lower() + '_sentiment_analysis.png')
    plt.close()  # Close the figure to avoid displaying it interactively

################# TASK 3

# Extract location information from the tweet data
location_df = tweet_df.select(explode(tweet_df.place.bounding_box.coordinates).alias("location"))

# Group by location and count the number of tweets
location_counts = location_df.groupBy("location").count()

# Sort the locations by tweet count in descending order
location_counts = location_counts.orderBy(col("count").desc())

# Show the top locations with the highest tweet counts
location_counts.show()

# Calculate the low and high thresholds dynamically based on the tweet counts
tweet_counts = location_counts.select("count").rdd.flatMap(lambda x: x).collect()
# Calculate the low threshold dynamically
low_threshold = 10  # Set a fixed low threshold

# Calculate the high threshold dynamically based on the maximum tweet count
max_tweet_count = max(tweet_counts)
high_threshold = np.log(max_tweet_count) 

# Define a color scale for severity levels (tweet counts)
color_scale = {
    "low": "green",
    "medium": "orange",
    "high": "red"
}

# Initialize the map centered around a location (e.g., world map)
m = folium.Map(location=[0, 0], zoom_start=2)

# Add markers for each location with the severity (tweet count) as the popup and color based on severity level
for row in location_counts.collect():
    location = row["location"]
    tweet_count = row["count"]
    severity = "low"
    if tweet_count >= high_threshold:
        severity = "high"
    elif tweet_count >= low_threshold:
        severity = "medium"
    color = color_scale[severity]
    # Calculate the marker radius based on the number of tweets
    radius = min(tweet_count / 500, 15)  # Limiting the maximum radius to 15
    folium.CircleMarker(location=[location[1][1], location[1][0]], radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7, popup=f"Tweet Count: {tweet_count}").add_to(m)

# Save the map as an HTML file
m.save("tweet_distribution_map.html")
