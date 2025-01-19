import argparse
from download_books import download_and_prepare_all, download_and_prepare_dataset
from json_to_parquet import convert_json_to_parquet
from analyze_shelves import analyze_shelves
from filter_books import filter_books
from transform_books import transform_books
import os

def run_complete_pipeline(openai_api_key: str = None):
    """Run the complete pipeline with default settings"""
    print("\n=== Step 1: Downloading and Preparing Data ===")
    successful_files = download_and_prepare_all()
    if not successful_files:
        print("Download failed. Stopping pipeline.")
        return
    print("Download step completed successfully.")
    
    print("\n=== Step 2: Converting to Parquet ===")
    convert_json_to_parquet("goodreads_books.json", "goodreads_books")
    
    print("\n=== Step 3: Creating Custom Genres Index ===")
    analyze_shelves(
        input_file="goodreads_books.parquet",
        output_file="classified_shelves.parquet",
        top_n=1000,
        openai_api_key=openai_api_key
    )
    
    print("\n=== Step 4: Filtering Books ===")
    filter_books(
        input_file="goodreads_books.parquet",
        output_file="filtered_books.parquet",
        min_reviews=1000
    )
    
    print("\n=== Step 5: Transforming Books ===")
    transform_books(
        input_file="filtered_books.parquet",
        output_file="enriched_books.parquet"
    )
    
    print("\nComplete pipeline executed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Goodreads Books Data Processing Pipeline')
    parser.add_argument('--step', type=str, choices=['download', 'convert', 'all'],
                      help='Which step to execute: download, convert, or all')
    parser.add_argument('--dataset', type=str, choices=['books', 'authors', 'genres', 'all'],
                      default='all', help='Which dataset to process')
    parser.add_argument('--openai-api-key', type=str,
                      help='OpenAI API key for genre classification')
    
    args = parser.parse_args()
    
    # If no arguments provided, run the complete pipeline
    if not args.step:
        openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OpenAI API key must be provided (--openai-api-key) or set in OPENAI_API_KEY environment variable")
            return
        run_complete_pipeline(openai_api_key)
        return
    
    # Otherwise, run specific steps as before
    if args.step == 'download' or args.step == 'all':
        print("\n=== Step 1: Downloading and Preparing Data ===")
        if args.dataset == 'all':
            successful_files = download_and_prepare_all()
        else:
            if download_and_prepare_dataset(args.dataset):
                successful_files = [args.dataset]
            else:
                successful_files = []
                
        if not successful_files:
            print("Download failed. Stopping pipeline.")
            return
        print("Download step completed successfully.")
    
    if args.step == 'convert' or args.step == 'all':
        print("\n=== Step 2: Converting to Parquet ===")
        if args.dataset in ['books', 'all']:
            input_file = "goodreads_books.json"
            output_file = "goodreads_books"
            convert_json_to_parquet(input_file, output_file)
        print("Conversion step completed.")

if __name__ == "__main__":
    main() 