import json
import pandas as pd
import ijson  # For memory-efficient JSON parsing
import os
import glob

def preprocess_record(record):
    """
    Preprocess each record to adjust the structure as needed.
    Convert popular_shelves and authors to simple arrays.
    """
    if 'popular_shelves' in record:
        record['popular_shelves'] = [shelf['name'] for shelf in record['popular_shelves']]
    if 'authors' in record:
        record['authors'] = [author['author_id'] for author in record['authors']]
    return record

def merge_parquet_files(base_path, final_path):
    """
    Merge all chunk files into a single parquet file
    """
    print("\nMerging parquet files...")
    # Get all chunk files
    chunk_files = glob.glob(f"{base_path}_chunk_*.parquet")
    
    if not chunk_files:
        print("No chunk files found to merge")
        return False
    
    # Read and concatenate all chunks
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
        
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Save final parquet file
    final_df.to_parquet(final_path)
    
    # Clean up chunk files
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    print(f"Merged {len(chunk_files)} chunks into {final_path}")
    print(f"Total records: {len(final_df)}")
    return True

def check_existing_chunks(base_path):
    """
    Check if chunk files exist and return their count
    """
    chunk_files = glob.glob(f"{base_path}_chunk_*.parquet")
    return len(chunk_files) > 0, chunk_files

def convert_json_to_parquet(json_path, parquet_path, chunk_size=100000):
    """
    Convert a large JSON file to Parquet format by reading in chunks.
    If chunk files already exist, goes directly to merging.
    """
    final_parquet_path = f"{parquet_path}.parquet"
    
    # Check if final parquet file already exists
    if os.path.exists(final_parquet_path):
        print(f"Final parquet file {final_parquet_path} already exists. Skipping conversion.")
        return True
        
    # Check for existing chunks
    chunks_exist, _ = check_existing_chunks(parquet_path)
    if chunks_exist:
        print("Found existing chunk files. Proceeding directly to merge...")
        return merge_parquet_files(parquet_path, final_parquet_path)
    
    print("Converting JSON to Parquet...")
    try:
        # Convert to Parquet in chunks
        chunk_index = 0
        with open(json_path, 'rb') as file:
            objects = ijson.items(file, '', multiple_values=True)
            chunk = []
            
            for i, obj in enumerate(objects):
                processed_obj = preprocess_record(obj)
                chunk.append(processed_obj)
                
                if len(chunk) >= chunk_size:
                    df = pd.json_normalize(chunk)  # Flatten nested JSON
                    chunk_file = f"{parquet_path}_chunk_{chunk_index}.parquet"
                    df.to_parquet(chunk_file, index=False)
                    print(f"Processed {i+1} records, saved to {chunk_file}")
                    chunk = []
                    chunk_index += 1
            
            # Write remaining records
            if chunk:
                df = pd.json_normalize(chunk)
                chunk_file = f"{parquet_path}_chunk_{chunk_index}.parquet"
                df.to_parquet(chunk_file, index=False)
                print(f"Processed remaining records, saved to {chunk_file}")

        # Merge all chunks into final parquet file
        success = merge_parquet_files(parquet_path, final_parquet_path)
        
        if success:
            print("\nConversion complete. Final parquet file saved at:", final_parquet_path)
            return True
        return False
        
    except Exception as e:
        print(f"Error converting JSON to Parquet: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "goodreads_books.json"
    output_file = "goodreads_books"
    convert_json_to_parquet(input_file, output_file) 