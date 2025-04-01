"""Reddit Sentiment"""


import praw
import pandas as pd

# Reddit API credentials
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

print(reddit.read_only)

def get_top_posts_about_coin(
    reddit,
    coin_keyword: str,
    end_datetime: datetime,
    hours: int = 24,
    subreddit_name: str = "Cryptocurrency",
    limit: int = 10
):
    start_ts = int((end_datetime - timedelta(hours=hours)).timestamp())
    end_ts   = int(end_datetime.timestamp())

    # CloudSearch syntax: 'timestamp:START..END AND KEYWORD'
    query = f"timestamp:{start_ts}..{end_ts} AND {coin_keyword}"

    subreddit = reddit.subreddit(subreddit_name)

    # Retrieve submissions via 'search()'
    # 'syntax="cloudsearch"' is crucial for timestamp-range queries.
    results = subreddit.search(query, syntax="cloudsearch", sort="new", limit=limit)

    posts_data = []
    for submission in results:
        # Even though we used timestamps, it’s good to double-check:
        if start_ts <= submission.created_utc <= end_ts:
            text_combined = (
                submission.title + " " + (submission.selftext or "")
            ).lower()
            if coin_keyword.lower() in text_combined:
                posts_data.append({
                    'title':        submission.title,
                    'selftext':     submission.selftext,
                    'score':        submission.score,
                    'created_utc':  submission.created_utc,
                    'url':          submission.url
                })

    df_posts = pd.DataFrame(posts_data)
    if not df_posts.empty:
        df_posts.sort_values(by='score', ascending=False, inplace=True)
        df_posts.reset_index(drop=True, inplace=True)
    return df_posts

import pandas as pd
from datetime import datetime, timedelta

def get_top_posts_about_coin(
    reddit,
    coin_keyword: str,
    end_datetime: datetime,
    hours: int = 24,
    subreddit_name: str = "Cryptocurrency",
    limit: int = 5
):
    """
    Pulls posts from `subreddit_name` that mention `coin_keyword`
    within the 24-hour window ending at `end_datetime`. Returns a
    DataFrame sorted by descending score (top posts).

    Parameters:
    -----------
    reddit : praw.Reddit
        An authenticated PRAW Reddit instance.
    coin_keyword : str
        The coin name or symbol to search for (e.g., 'Bitcoin' or 'BTC').
    end_datetime : datetime
        The (historical) end of your 24-hour window (UTC preferred).
    hours : int
        How many hours before `end_datetime` to include (default=24).
    subreddit_name : str
        The subreddit to search (default="Cryptocurrency").
    limit : int
        How many posts to pull from the "new" listing before filtering.

    Returns:
    --------
    pd.DataFrame
        Columns: ['title', 'selftext', 'score', 'created_utc', 'url'].
        Sorted by 'score' in descending order.
    """
    start_datetime = end_datetime - timedelta(hours=hours)
    start_ts = start_datetime.timestamp()
    end_ts = end_datetime.timestamp()

    print(start_datetime, end_datetime)

    # We'll collect data in a list of dicts, then convert to DataFrame.
    posts_data = []

    subreddit = reddit.subreddit(subreddit_name)

    # Retrieve up to `limit` recent posts via subreddit.new()
    for submission in subreddit.new(limit=limit):
        created_utc = submission.created_utc

        # Skip if the post is created after our end window
        if created_utc > end_ts:
            continue

        # Break if the post is older than our start window
        if created_utc < start_ts:
            break

        # Check if coin keyword is in title or selftext
        text_combined = (submission.title + " " + (submission.selftext or "")).lower()
        if coin_keyword.lower() in text_combined:
            posts_data.append({
                'title':        submission.title,
                'selftext':     submission.selftext,
                'score':        submission.score,
                'created_utc':  submission.created_utc,
                'url':          submission.url
            })

    # Convert to DataFrame
    df_posts = pd.DataFrame(posts_data)

    # Sort descending by score to get "top" posts
    if not df_posts.empty:
        df_posts.sort_values(by='score', ascending=False, inplace=True)
        df_posts.reset_index(drop=True, inplace=True)

    return df_posts

test = get_top_posts_about_coin(reddit, "bitcoin", datetime.now())
print(test)

"""###Get posts from CSV"""

#mount drive to access files


import pandas as pd
from datetime import datetime, timedelta


def get_top_posts_about_coin_from_csv(
    csv_path: str,
    coin_keyword: str,
    end_datetime: datetime,
    hours: int = 24,
    limit: int = 10
):
    """
    """
    df = pd.read_csv(csv_path)

    start_datetime = end_datetime - timedelta(hours=hours)
    start_ts = start_datetime.timestamp()
    end_ts = end_datetime.timestamp()

    print("Time window:")
    print(f"   Start: {start_datetime} ({start_ts})")
    print(f"   End:   {end_datetime}   ({end_ts})")

    print(df.columns)

    mask_time = (df['created'] >= start_ts) & (df['created'] <= end_ts)
    df_window = df[mask_time].copy()

    # 3) Check if coin_keyword is in the title or selftext (case-insensitive)
    coin_kw_lower = coin_keyword.lower()

    # Ensure no NaNs
    df_window['title'] = df_window['title'].fillna('')
    df_window['selftext'] = df_window['selftext'].fillna('')

    mask_coin = df_window.apply(
        lambda row: coin_kw_lower in (row['title'] + ' ' + row['selftext']).lower(),
        axis=1
    )
    df_coin = df_window[mask_coin]

    # 4) Sort by descending score
    df_coin.sort_values(by='score', ascending=False, inplace=True)

    # 5) Take top_n
    df_top = df_coin.head(limit).copy()
    df_top.reset_index(drop=True, inplace=True)

    return df_top

#test
df = get_top_posts_about_coin_from_csv("/content/drive/MyDrive/229_data_filtered.csv", "bitcoin", datetime(2022, 1, 23))
print(df)

"""###Sentiment Analysis

"""

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")  # default distilBERT model

def analyze_post_sentiment(text):
    # This returns a list of predictions, each with {label: "POSITIVE"/"NEGATIVE", score: float}
    result = sentiment_pipeline(text[:512])  # trunc to 512 tokens for quick demo
    return result[0]['label'], result[0]['score']

#Get posts
df_reddit = get_top_posts_about_coin_from_csv("/content/drive/MyDrive/229_data_filtered.csv", "bitcoin", datetime(2022, 1, 23))
print(df_reddit.size)

def compute_daily_sentiment_score(df_reddit):

  sentiments = []
  for idx, row in df_reddit.iterrows():
      combined_text = f"{row['title']} {row['selftext']}"
      label, score = analyze_post_sentiment(combined_text)
      sentiments.append({'label': label, 'score': score})

  df_reddit['sentiment_label'] = [s['label'] for s in sentiments]
  df_reddit['sentiment_score'] = [s['score'] for s in sentiments]

  mean_positive = df_reddit[df_reddit['sentiment_label'] == 'POSITIVE']['sentiment_score'].mean()
  mean_negative = df_reddit[df_reddit['sentiment_label'] == 'NEGATIVE']['sentiment_score'].mean()

  # Quick, naive daily sentiment index:
  if pd.isna(mean_positive):
      mean_positive = 0
  if pd.isna(mean_negative):
      mean_negative = 0
  daily_sentiment_index = mean_positive - mean_negative
  return daily_sentiment_index

def get_sentiment_scores(coin_keyword, start_date, end_date, limit=5):
    #for each day within [start, end date]
    #Pull the LIMIT # of reddit posts for that coin
    #Run sentiment analysis for each post
    #And then aggregate that into a daily mean sentiment score for each coin
    current_date = start_date
    sentiment_data = []

    while current_date <= end_date:
        for coin in coin_keyword:
            df_posts = get_top_posts_about_coin_from_csv(
                csv_path="/content/drive/MyDrive/229_data_filtered.csv",
                coin_keyword=coin,
                end_datetime=current_date,
                hours=24,
                limit=limit
            )

            daily_sentiment = compute_daily_sentiment_score(df_posts)

            sentiment_data.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Coin": coin,
                "Daily Sentiment Score": daily_sentiment
            })

        # Move to the next day
        current_date += timedelta(days=1)

    # Convert results to DataFrame
    df_daily_sentiment = pd.DataFrame(sentiment_data)

    return df_daily_sentiment

"""Sentiment Pipeline Test"""

print(get_sentiment_scores(["bitcoin", "ethereum"], datetime(2022, 1, 23) - timedelta(days=5), datetime(2022, 1, 23), limit=10))