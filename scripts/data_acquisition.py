from huggingface_hub import hf_hub_download

print("Starting dataset download...")

path = hf_hub_download(
    repo_id="sinhal/Indian_Supreme_Court_Judgments",
    filename="laww_dataset.jsonl",
    repo_type="dataset"
)

print("Dataset downloaded successfully!")
print("Local path:")
print(path)