import argparse
from download_books import download_and_prepare_all, download_and_prepare_dataset
from json_to_parquet import convert_json_to_parquet

def main():
    parser = argparse.ArgumentParser(description='Goodreads Books Data Processing Pipeline')
    parser.add_argument('--step', type=str, choices=['download', 'convert', 'all'],
                      help='Which step to execute: download, convert, or all')
    parser.add_argument('--dataset', type=str, choices=['books', 'authors', 'genres', 'all'],
                      default='all', help='Which dataset to process')
    
    args = parser.parse_args()
    
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