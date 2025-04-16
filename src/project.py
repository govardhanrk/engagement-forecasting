from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from geopy.geocoders import Nominatim

# Initialize Spark session
spark = SparkSession.builder \
    .appName("COVID-19 Twitter Analysis") \
    .getOrCreate()

# Load JSON data into DataFrame
tweet_df = spark.read.json("tweets.json")

# Task 1: Mapping the COVID-19 Conversation

# Extract location information
extract_location_udf = udf(lambda user: user['location'] if user is not None else None, StringType())
tweet_df = tweet_df.withColumn("location", extract_location_udf(col("user")))

# Geocoding
geolocator = Nominatim(user_agent="twitter_analysis")
geocode_udf = udf(lambda location: geolocator.geocode(location).point if location else None)
tweet_df = tweet_df.withColumn("coordinates", geocode_udf(col("location")))

# Task 2: Deciphering the Vaccine Sentiment (using pre-trained model)

# Assume sentiment analysis function is defined elsewhere
sentiment_udf = udf(lambda text: analyze_sentiment(text), StringType())
tweet_df = tweet_df.withColumn("sentiment", sentiment_udf(col("full_text")))

# Task 3: Unveiling Variant Perspectives

# Extract variant mentions
variant_keywords = ["Delta", "Omicron"]
extract_variant_udf = udf(lambda text: any(keyword in text for keyword in variant_keywords), StringType())
tweet_df = tweet_df.withColumn("mentions_variant", extract_variant_udf(col("full_text")))

# Task 4: Forecasting Vaccine Uptake (sample code)

# Feature Engineering
# Assuming features include timestamps, user demographics, and vaccine-related keywords

# Model Selection
# Assuming logistic regression model is chosen
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["feature1", "feature2", ...], outputCol="features")
training_data = assembler.transform(tweet_df)

lr = LogisticRegression(featuresCol='features', labelCol='vaccine_uptake')
lr_model = lr.fit(training_data)

# Prediction
predictions = lr_model.transform(training_data)

# Task 5: Identifying Influential Voices

# Calculate user engagement metrics
engagement_metrics = tweet_df.groupBy("user").agg(
    {"retweet_count": "sum", "favorite_count": "sum", "reply_count": "sum"}
)

# Network Analysis
# Assuming graph construction and analysis are done separately

# Save results to files or databases for further analysis or visualization
# For example, saving DataFrame to CSV files
tweet_df.write.csv("tweets_processed.csv")

# Stop Spark session
spark.stop()

############################################
# Importing necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

# Initializing Spark session
spark = SparkSession.builder \
    .appName("COVID-19 Twitter Analysis") \
    .getOrCreate()

# Task 1: Mapping the COVID-19 Conversation

# Load JSON data into DataFrame
tweets_df = spark.read.json("covid19_tweets.json")

# Extracting location information
extract_location = udf(lambda user: user.get("location", ""), StringType())
tweets_df = tweets_df.withColumn("location", extract_location(col("user")))

# Geocoding locations to coordinates
geolocator = Nominatim(user_agent="covid_analysis")
geocode_udf = udf(lambda loc: geolocator.geocode(loc).point if loc else None)
tweets_df = tweets_df.withColumn("coordinates", geocode_udf(col("location")))

# Aggregating and analyzing tweet volumes by location
location_counts = tweets_df.groupBy("location").count().orderBy(col("count").desc()).limit(10)

# Displaying top locations with high tweet volumes
location_counts.show()

# Task 2: Deciphering the Vaccine Sentiment

# Assuming sentiment analysis is not available, we'll simply visualize sentiment based on keywords

# Preprocessing tweet text for sentiment analysis (removing noise, stopwords, etc.)
tweets_df = tweets_df.withColumn("clean_text", <add preprocessing function>)

# Identifying tweets mentioning vaccines
vaccine_tweets_df = tweets_df.filter(col("clean_text").contains("vaccine"))

# Visualizing sentiment distribution
vaccine_sentiment = vaccine_tweets_df.groupBy("sentiment").count().toPandas()

plt.bar(vaccine_sentiment["sentiment"], vaccine_sentiment["count"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution of COVID-19 Vaccine Tweets")
plt.show()

# Task 3: Unveiling Variant Perspectives

# Assuming variant identification is not available, we'll simply visualize sentiment based on COVID-19 variants

# Preprocessing tweet text for variant identification (removing noise, stopwords, etc.)
tweets_df = tweets_df.withColumn("clean_text", <add preprocessing function>)

# Identifying tweets mentioning COVID-19 variants
variant_tweets_df = tweets_df.filter(col("clean_text").contains("delta") | col("clean_text").contains("omicron"))

# Visualizing sentiment towards variants
variant_sentiment = variant_tweets_df.groupBy("variant", "sentiment").count().orderBy(col("count").desc()).limit(10)

# Displaying sentiment towards variants
variant_sentiment.show()

# Task 4: Forecasting Vaccine Uptake

# Assuming predictive modeling for vaccine uptake is not available

# Task 5: Identifying Influential Voices

# Assuming user engagement metrics and network analysis are not available

# Stop Spark session
spark.stop()
