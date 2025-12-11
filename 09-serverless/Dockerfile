FROM agrigorev/model-2025-hairstyle:v1

# Install dependencies for preprocessing and ONNX inference
RUN pip install --no-cache-dir numpy pillow onnxruntime

# Copy your Lambda handler
COPY lambda_function.py .

# Entry point for AWS Lambda
CMD ["lambda_function.lambda_handler"]