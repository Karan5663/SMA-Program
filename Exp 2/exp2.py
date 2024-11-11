import requests
import pandas as pd
from textblob import TextBlob
import os

# Fetch API key securely from environment variable
api_key = os.getenv("AIzaSyAoEr28aXt-7AigbzfDw18Y6dYIijIGjcY")  # Ensure that YOUTUBE_API_KEY environment variable is set

# Alternatively, hardcode your API key (if not using environment variables)
# api_key = "YOUR_NEW_API_KEY"  # Replace this with your actual API key

# Ensure that the API key is set
if not api_key:
    print("API key not set. Please ensure the environment variable YOUTUBE_API_KEY is configured.")
    exit()

# Replace with the actual video ID (ensure this video is public and available)
video_id = "https://youtu.be/VKf6NF0OD5A"  # Example video ID

# Function to retrieve video information
def get_video_info(video_id, api_key):
    video_info_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    try:
        response = requests.get(video_info_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve video info, status code {response.status_code}")
            print("Response:", response.text)  # Detailed error message from API
            return None
    except Exception as e:
        print(f"Error while fetching video info: {e}")
        return None

# Function to retrieve video comments
def get_video_comments(video_id, api_key):
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}"
    comments = []
    next_page_token = None

    try:
        while True:
            url = comments_url + (f"&pageToken={next_page_token}" if next_page_token else "")
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                comments.extend([item["snippet"]["topLevelComment"]["snippet"]["textOriginal"] for item in data["items"]])
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break  # No more pages of comments
            else:
                print(f"Failed to fetch comments, status code {response.status_code}")
                print("Response:", response.text)
                break
    except Exception as e:
        print(f"Error while fetching comments: {e}")

    return comments

# Function to analyze sentiment of comments using TextBlob
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
    print("Video Info:", video_info_data)
else:
    print("Failed to retrieve video info.")

# Fetch video comments
comments_data = get_video_comments(video_id, api_key)
if comments_data:
    # Analyze sentiment of comments
    comment_list = []
    sentiment_list = []
    
    for comment in comments_data:
        sentiment = get_comment_sentiment(comment)
        comment_list.append(comment)
        sentiment_list.append(sentiment)
        print(f"Comment: {comment} | Sentiment: {sentiment}")
    
    # Create a DataFrame and save to CSV
    sentiment_df = pd.DataFrame({"Comments": comment_list, "Sentiment": sentiment_list})
    sentiment_df.to_csv("YouTube_Comments_Sentiment.csv", index=False)
    print("Comments sentiment saved to CSV.")
else:
    print("Failed to retrieve comments.")
