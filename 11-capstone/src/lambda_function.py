import json

from .predictor import predict


def handler(event, context=None):
    """
    Supports two invocation styles:

    1) Direct payload:
       {"type":"TV", ...}

    2) Body-wrapped payload (API Gateway-ish):
       {"body":"{...json...}"} or {"body":{...}}
    """
    payload = event
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        payload = json.loads(body) if isinstance(body, str) else body

    try:
        out = predict(payload)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(out),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
