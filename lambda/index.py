# lambda/index.py
import json
import os
import re
import urllib3  # 標準ライブラリでOK
from botocore.exceptions import ClientError

# Lambda内からのリージョン抽出（使わないが残す場合）
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"

# FastAPIサーバーのURL（環境変数で指定するのが推奨）
FASTAPI_URL = os.environ.get("FASTAPI_ENDPOINT", "https://80e4-34-124-200-155.ngrok-free.app")

# HTTPクライアント
http = urllib3.PoolManager()

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        #Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])

        # FastAPIに送信するペイロード
        payload = {
            "message": message,
            "conversationHistory": conversation_history
        }

        print("Sending request to FastAPI:", FASTAPI_URL)

        # FastAPIにPOSTリクエストを送信
        response = http.request(
            "POST",
            FASTAPI_URL,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        print("FastAPI response status:", response.status)
        print("FastAPI response body:", response.data.decode())

        if response.status != 200:
            raise Exception(f"FastAPI responded with status {response.status}")

        response_body = json.loads(response.data.decode())

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": response_body.get("response"),
                "conversationHistory": response_body.get("conversationHistory")
            })
        }

    except Exception as error:
        print("Error:", str(error))

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
