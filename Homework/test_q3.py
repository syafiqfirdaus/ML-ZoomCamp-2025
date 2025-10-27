import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

print("Pipeline loaded successfully!")
print(f"Pipeline type: {type(pipeline)}")

# Score the record for Question 3
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict probability
probability = pipeline.predict_proba([client])[0, 1]

print(f"\nQuestion 3:")
print(f"Probability that this lead will convert: {probability:.3f}")
