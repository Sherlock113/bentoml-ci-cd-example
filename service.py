from __future__ import annotations
import bentoml

with bentoml.importing():
    from transformers import pipeline


EXAMPLE_INPUT = "this is a test to see if the code update works"


my_image = bentoml.images.Image(python_version="3.11") \
        .python_packages("torch", "transformers")


@bentoml.service(
    image=my_image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class Summarization:
    # Define the Hugging Face model as a class variable
    model_path = bentoml.models.HuggingFaceModel("sshleifer/distilbart-cnn-12-6")

    def __init__(self) -> None:
        # Load model into pipeline
        self.pipeline = pipeline('summarization', model=self.model_path)
    
    @bentoml.api
    def summarize(self, text: str = EXAMPLE_INPUT) -> str:
        result = self.pipeline(text)
        return f"Hello world! Here's your summary: {result[0]['summary_text']}"
