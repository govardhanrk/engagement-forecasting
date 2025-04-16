# COVID-19 Twitter Analysis

This project analyzes COVID-19-related tweets to uncover insights such as sentiment analysis, variant mentions, vaccine uptake forecasting, and influential voices. The analysis leverages big data techniques and machine learning to provide actionable insights.

## Project Objectives
- **Sentiment Analysis**: Understand public sentiment regarding COVID-19.
- **Variant Mentions**: Track mentions of COVID-19 variants like Delta and Omicron.
- **Geospatial Analysis**: Visualize tweet distribution and identify hotspots.
- **Forecasting**: Predict vaccine uptake trends using machine learning models.
- **Influential Voices**: Identify key influencers driving engagement.

## Project Structure
```
engagement-forecasting/
├── BigDataPoster.pdf       # Project poster summarizing key findings
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Directory containing datasets
│   └── tweets.json.zip     # Compressed dataset of tweets
├── src/                    # Source code for the project
│   ├── project.py          # Main script for analysis
│   ├── project_tasks.py    # Additional tasks and utilities
```

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/covid-twitter-analysis.git
   cd covid-twitter-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Navigate to the `data/` directory and unzip the dataset:
     ```bash
     cd data
     unzip tweets.json.zip
     ```

4. **Run the analysis**:
   ```bash
   python src/project.py
   ```

## Results and Insights
- **Sentiment Analysis**: Majority of tweets express neutral sentiment, with spikes in positive sentiment during vaccine rollouts.
- **Variant Mentions**: Delta variant was the most discussed, followed by Omicron.
- **Geospatial Analysis**: High tweet activity observed in urban areas.
- **Forecasting**: Predicted vaccine uptake aligns with actual trends.
- **Influential Voices**: Identified key influencers with high engagement metrics.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: TextBlob, Matplotlib, Pandas, NumPy, Scikit-learn
- **Big Data Tools**: PySpark
- **Visualization**: Folium, WordCloud

## License
This project is licensed under the MIT License.