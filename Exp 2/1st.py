import requests
import pandas as pd
from textblob import TextBlob

# Set video ID and API key
video_id = "PbgUg_aBHf4"
api_key = "AIzaSyAoEr28aXt-7AigbzfDw18Y6dYIijIGjcY"  # Replace with your actual YouTube Data API key

# Function to retrieve video information
def get_video_info(video_id, api_key):
    video_info_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    response = requests.get(video_info_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to retrieve video comments
def get_video_comments(video_id, api_key):
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}"
    response = requests.get(comments_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to analyze sentiment of comments
def get_comment_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Fetch video information
video_info_data = get_video_info(video_id, api_key)
if video_info_data:
    print(video_info_data)
else:
    print("Failed to retrieve video info.")

# Fetch video comments
comments_data = get_video_comments(video_id, api_key)
if comments_data and "items" in comments_data:
    comments = [item["snippet"]["topLevelComment"]["snippet"]["textOriginal"] for item in comments_data["items"]]

    # Analyze sentiment
    comment_list = []
    sentiment_list = []
    
    for comment in comments:
        sentiment = get_comment_sentiment(comment)
        comment_list.append(comment)
        sentiment_list.append(sentiment)
        print(f"{comment} : {sentiment}")
    
    # Create a DataFrame and save to CSV
    sentiment_df = pd.DataFrame({"Comments": comment_list, "Sentiment": sentiment_list})
    sentiment_df.to_csv("YouTube_Comments_Sentiment.csv", index=False)
    print("Comments sentiment saved to CSV.")
else:
    print("Failed to retrieve comments.")