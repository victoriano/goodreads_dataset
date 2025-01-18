import requests
import gzip
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

# Define the files to download with their URLs
DATASET_FILES = {
    'books': {
        'url': 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz',
        'gz_name': 'goodreads_books.json.gz',
        'output_name': 'goodreads_books.json'
    },
    'authors': {
        'url': 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz',
        'gz_name': 'goodreads_book_authors.json.gz',
        'output_name': 'goodreads_book_authors.json'
    },
    'genres': {
        'url': 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz',
        'gz_name': 'goodreads_book_genres_initial.json.gz',
        'output_name': 'goodreads_book_genres_initial.json'
    }
}

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from URL showing a progress bar
    Returns True if successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

def decompress_gz(gz_path: str, output_path: str) -> bool:
    """
    Decompress a .gz file
    Returns True if successful, False otherwise
    """
    try:
        print(f"Decompressing {gz_path}...")
        with gzip.open(gz_path, 'rb') as gz_file:
            with open(output_path, 'wb') as output_file:
                output_file.write(gz_file.read())
        print(f"Decompressed file saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error decompressing file: {str(e)}")
        return False

def download_and_prepare_dataset(file_key: str) -> bool:
    """
    Download and decompress a specific dataset file
    """
    if file_key not in DATASET_FILES:
        print(f"Unknown file key: {file_key}")
        return False
        
    file_info = DATASET_FILES[file_key]
    try:
        # Download .gz file
        if not download_file(file_info['url'], file_info['gz_name']):
            return False
        
        # Decompress .gz file
        if not decompress_gz(file_info['gz_name'], file_info['output_name']):
            return False
        
        # Clean up .gz file
        os.remove(file_info['gz_name'])
        print(f"Cleaned up compressed file {file_info['gz_name']}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_key}: {str(e)}")
        return False

def download_and_prepare_all() -> List[str]:
    """
    Download and decompress all dataset files
    Returns list of successfully processed files
    """
    successful_files = []
    failed_files = []
    
    for file_key in DATASET_FILES:
        print(f"\nProcessing {file_key} dataset...")
        if download_and_prepare_dataset(file_key):
            successful_files.append(file_key)
        else:
            failed_files.append(file_key)
    
    # Print summary
    print("\n=== Download Summary ===")
    if successful_files:
        print("Successfully processed:")
        for file in successful_files:
            print(f"- {DATASET_FILES[file]['output_name']}")
    
    if failed_files:
        print("\nFailed to process:")
        for file in failed_files:
            print(f"- {DATASET_FILES[file]['output_name']}")
            
    return successful_files

if __name__ == "__main__":
    download_and_prepare_all() 