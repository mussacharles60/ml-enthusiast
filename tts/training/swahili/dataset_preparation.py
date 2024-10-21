import os
from typing import List, Dict
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, Audio
from huggingface_hub import HfApi, create_repo


def create_dataset_entries(filtered_dataset) -> List[Dict]:
    """Create dataset entries based on the filtered dataset."""
    data = []
    for i, item in enumerate(filtered_dataset):
        entry = {
            'line_id': f"SW{i:04d}",
            'audio': item['path'],
            'text': item['sentence'],
            'speaker_id': item['client_id'],

        }
        data.append(entry)
    return data


def create_dataset_entries_from_disk(clips_dir, filtered_dataset) -> List[Dict]:
    """Create dataset entries based on the filtered dataset."""
    data = []
    for i, item in filtered_dataset.iterrows():
        entry = {
            'line_id': f"SW{i:04d}",
            # Correct audio file path
            'audio': os.path.join(clips_dir, item['path']),
            'text': item['sentence'],
            'speaker_id': item['client_id'],
        }
        data.append(entry)
    return data


def upload_to_huggingface(dataset: Dataset, repo_id: str) -> None:
    """Upload the dataset to Hugging Face."""
    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="dataset")
        print("Repository created successfully.")
    except Exception as e:
        print(f"Repository creation failed or already exists: {e}")

    dataset.push_to_hub(repo_id)
    print("Dataset uploaded successfully!")


def get_dataset(client_id: str):
    # Load the Swahili dataset
    original_dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0", "sw")

    # Filter the dataset for the specific client_id
    filtered_dataset = original_dataset.filter(
        lambda example: example['client_id'] == client_id)

    # Print the number of rows after filtering
    print(
        f"Number of rows for client ID {client_id}: {len(filtered_dataset['train'])}")

    # Create dataset entries
    data = create_dataset_entries(filtered_dataset["train"])

    return data


def get_dataset_from_disk(client_id: str):
    # Define the path to the extracted Swahili dataset
    data_dir = "E:/Downloads/cv-corpus-17.0-2024-03-15-sw/cv-corpus-17.0-2024-03-15/sw"

    # Path to the audio clips folder
    clips_dir = os.path.join(data_dir, "clips")

    # Path to the TSV file containing metadata (e.g., train.tsv)
    # Can be 'dev.tsv', 'test.tsv', etc.
    train_tsv_path = os.path.join(data_dir, "train.tsv")

    # Step 1: Load the TSV metadata file

    metadata = pd.read_csv(train_tsv_path, sep='\t')

    # Step 2: Filter the dataset for the specific client_id
    filtered_dataset = metadata[metadata['client_id'] == client_id]

    # Print the number of rows after filtering
    print(f"Number of rows for client ID {client_id}: {len(filtered_dataset)}")

    # Create dataset entries
    data = create_dataset_entries_from_disk(clips_dir, filtered_dataset)

    return data


def create_and_upload_dataset(repo_id: str, client_id: str) -> None:
    """
    Create a dataset from the Mozilla Common Voice dataset for a specific client_id and upload it to Hugging Face.
    """

    # # get dataset from huggingface
    # data = get_dataset(client_id)

    # get dataset from disk
    data = get_dataset_from_disk(client_id)

    # Create Dataset
    dataset = Dataset.from_dict({
        'line_id': [item['line_id'] for item in data],
        'audio': [item['audio'] for item in data],
        'text': [item['text'] for item in data],
        'speaker_id': [item['speaker_id'] for item in data],
    })

    # Cast the audio column to Audio type
    # Adjust sampling rate if needed
    dataset = dataset.cast_column("audio", Audio(sampling_rate=48000))

    # Upload to Hugging Face
    upload_to_huggingface(dataset, repo_id)


# client_id = "052c5091df7681302a2117b2d21db1540c2156f5254ebe9876a7d0146588eab582e11cb47761a18f84200a510a5386bdf024374f76113cd15fe1cc8d7b9fcf0b"
client_id = "fe3befae02733265c3fc953eb67840c57d970340a76386ffda9ab3226d31e376790d7eddefde5f434647687e6136c44e50513edebca32377799b15363919310d"
create_and_upload_dataset("mcv-sw-female-dataset", client_id)
