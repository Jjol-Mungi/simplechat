# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError

#追加モジュール
import torch
from transformers import pipeline
import time
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import nest_asyncio
from pyngrok import ngrok



#追加コード~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app = FastAPI()

# リクエストの形式を定義
class InputData(BaseModel):
    message: str

# 直接プロンプトを使用した簡略化されたリクエスト
class SimpleGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    response_time: float

model = None

MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

def load_model():
    """推論用のLLMモデルを読み込む"""
    global model  # グローバル変数を更新するために必要
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用デバイス: {device}")
        pipe = pipeline(
            "text-generation",
            model=config.MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        print(f"モデル '{config.MODEL_ID}' の読み込みに成功しました")
        model = pipe  # グローバル変数を更新
        return pipe
    except Exception as e:
        error_msg = f"モデル '{config.MODEL_ID}' の読み込みに失敗: {e}"
        print(error_msg)
        traceback.print_exc()  # 詳細なエラー情報を出力
        return None

def load_model_task():
    """モデルを読み込むバックグラウンドタスク"""
    global model
    print("load_model_task: モデルの読み込みを開始...")
    # load_model関数を呼び出し、結果をグローバル変数に設定
    loaded_pipe = load_model()
    if loaded_pipe:
        model = loaded_pipe  # グローバル変数を更新
        print("load_model_task: モデルの読み込みが完了しました。")
    else:
        print("load_model_task: モデルの読み込みに失敗しました。")

print("FastAPIエンドポイントを定義しました。")

# --- FastAPIエンドポイント定義 ---
@app.on_event("startup")
async def startup_event():
    """起動時にモデルを初期化"""
    load_model_task()  # バックグラウンドではなく同期的に読み込む
    if model is None:
        print("警告: 起動時にモデルの初期化に失敗しました")
    else:
        print("起動時にモデルの初期化が完了しました。")

@app.get("/")
async def root():
    """基本的なAPIチェック用のルートエンドポイント"""
    return {"status": "ok", "message": "Local LLM API is runnning"}

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    global model
    if model is None:
        return {"status": "error", "message": "No model loaded"}

    return {"status": "ok", "model": config.MODEL_ID}

# 簡略化されたエンドポイント
@app.post("/generate", response_model=GenerationResponse)
async def generate_simple(request: SimpleGenerationRequest):
    """単純なプロンプト入力に基づいてテキストを生成"""
    global model

    if model is None:
        print("generateエンドポイント: モデルが読み込まれていません。読み込みを試みます...")
        load_model_task()  # 再度読み込みを試みる
        if model is None:
            print("generateエンドポイント: モデルの読み込みに失敗しました。")
            raise HTTPException(status_code=503, detail="モデルが利用できません。後でもう一度お試しください。")

    try:
        start_time = time.time()
        print(f"シンプルなリクエストを受信: prompt={request.prompt[:100]}..., max_new_tokens={request.max_new_tokens}")  # 長いプロンプトは切り捨て

        # プロンプトテキストで直接応答を生成
        print("モデル推論を開始...")
        outputs = model(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        print("モデル推論が完了しました。")

        # アシスタント応答を抽出
        assistant_response = extract_assistant_response(outputs, request.prompt)
        print(f"抽出されたアシスタント応答: {assistant_response[:100]}...")  # 長い場合は切り捨て

        end_time = time.time()
        response_time = end_time - start_time
        print(f"応答生成時間: {response_time:.2f}秒")

        return GenerationResponse(
            generated_text=assistant_response,
            response_time=response_time
        )

    except Exception as e:
        print(f"シンプル応答生成中にエラーが発生しました: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"応答の生成中にエラーが発生しました: {str(e)}")


class Config:
    def __init__(self, model_name=MODEL_ID):
        self.MODEL_ID = model_name

config = Config(MODEL_ID)



#公開urlの作成
def run_with_ngrok(port=8501):
    """ngrokでFastAPIアプリを実行"""
    nest_asyncio.apply()

    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        print("Ngrok認証トークンが'NGROK_TOKEN'環境変数に設定されていません。")
        try:
            print("Colab Secrets(左側の鍵アイコン)で'NGROK_TOKEN'を設定することをお勧めします。")
            ngrok_token = input("Ngrok認証トークンを入力してください (https://dashboard.ngrok.com/get-started/your-authtoken): ")
        except EOFError:
            print("\nエラー: 対話型入力が利用できません。")
            print("Colab Secretsを使用するか、ノートブックセルで`os.environ['NGROK_TOKEN'] = 'あなたのトークン'`でトークンを設定してください")
            return

    if not ngrok_token:
        print("エラー: Ngrok認証トークンを取得できませんでした。中止します。")
        return

    try:
        ngrok.set_auth_token(ngrok_token)

        # 既存のngrokトンネルを閉じる
        try:
            tunnels = ngrok.get_tunnels()
            if tunnels:
                print(f"{len(tunnels)}個の既存トンネルが見つかりました。閉じています...")
                for tunnel in tunnels:
                    print(f"  - 切断中: {tunnel.public_url}")
                    ngrok.disconnect(tunnel.public_url)
                print("すべての既存ngrokトンネルを切断しました。")
            else:
                print("アクティブなngrokトンネルはありません。")
        except Exception as e:
            print(f"トンネル切断中にエラーが発生しました: {e}")
            # エラーにもかかわらず続行を試みる

        # 新しいngrokトンネルを開く
        print(f"ポート{port}に新しいngrokトンネルを開いています...")
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        print("---------------------------------------------------------------------")
        print(f"✅ 公開URL:   {public_url}")
        print(f"📖 APIドキュメント (Swagger UI): {public_url}/docs")
        print("---------------------------------------------------------------------")
        print("(APIクライアントやブラウザからアクセスするためにこのURLをコピーしてください)")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # ログレベルをinfoに設定

    except Exception as e:
        print(f"\n ngrokまたはUvicornの起動中にエラーが発生しました: {e}")
        traceback.print_exc()
        # エラー後に残る可能性のあるngrokトンネルを閉じようとする
        try:
            print("エラーにより残っている可能性のあるngrokトンネルを閉じています...")
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                ngrok.disconnect(tunnel.public_url)
            print("ngrokトンネルを閉じました。")
        except Exception as ne:
            print(f"ngrokトンネルのクリーンアップ中に別のエラーが発生しました: {ne}")
run_with_ngrok(port=8501)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Nova Liteモデル用のリクエストペイロードを構築
        # 会話履歴を含める
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                bedrock_messages.append({
                    "role": "assistant", 
                    "content": [{"text": msg["content"]}]
                })
        
        # invoke_model用のリクエストペイロード
        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))
        
        # invoke_model APIを呼び出し
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # レスポンスを解析
        response_body = json.loads(response['body'].read())
        print("Bedrock response:", json.dumps(response_body, default=str))
        
        # 応答の検証
        if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
            raise Exception("No response content from the model")
        
        # アシスタントの応答を取得
        assistant_response = response_body['output']['message']['content'][0]['text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
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
                "response": assistant_response,
                "conversationHistory": messages
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
