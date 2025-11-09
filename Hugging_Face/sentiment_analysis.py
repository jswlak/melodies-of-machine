from transformers import pipeline

# Load a pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# Run inference
result = classifier("I love learning about AI with Hugging Face!")
print(result)
