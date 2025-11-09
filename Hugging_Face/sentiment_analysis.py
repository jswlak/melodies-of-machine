from transformers import pipeline

# Load a pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# Run inference
result = classifier("I love learning about AI with Hugging Face!")
print(result)





#Outpt 
#[{'label': 'POSITIVE', 'score': 0.9997183680534363}]