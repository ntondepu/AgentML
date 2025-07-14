import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "xoxb-your-token")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#general")

client = WebClient(token=SLACK_BOT_TOKEN)

def send_slack_message(text):
    try:
        response = client.chat_postMessage(channel=SLACK_CHANNEL, text=text)
        return response["ok"]
    except SlackApiError as e:
        print(f"Slack API error: {e.response['error']}")
        return False
