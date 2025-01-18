import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def load_authors_mapping(authors_file: str = "goodreads_book_authors.json") -> Dict[str, str]:
    """
    Load authors data and create a mapping from author_id to author name
    """
    print(f"Loading authors from {authors_file}...")
    authors_map = {}
    
    with open(authors_file, 'r') as f:
        for line in f:
            author = json.loads(line)
            authors_map[author['author_id']] = author.get('name', 'Unknown')
    
    print(f"Loaded {len(authors_map):,} authors")
    return authors_map

def load_genres_mapping(genres_file: str = "goodreads_book_genres_initial.json") -> Dict[str, List[str]]:
    """
    Load genres data and create a mapping from book_id to list of genres
    """
    print(f"Loading genres from {genres_file}...")
    genres_map = {}
    
    with open(genres_file, 'r') as f:
        for line in f:
            book_genres = json.loads(line)
            book_id = book_genres['book_id']
            # Filter genres with a certain confidence threshold
            genres = [genre for genre, score in book_genres['genres'].items() 
                     if float(score) > 0.3]  # Adjustable threshold
            if genres:
                genres_map[book_id] = genres
    
    print(f"Loaded genres for {len(genres_map):,} books")
    return genres_map

def parse_author_ids(author_str):
    """Convert any author string/array format to a list of string IDs"""
    try:
        # Handle null values
        if pd.isna(author_str):
            return []
            
        # Handle numpy arrays - convert directly to list of strings
        if isinstance(author_str, np.ndarray):
            return [str(x).strip().strip('"\'') for x in author_str]
            
        # Handle lists - convert elements to strings
        if isinstance(author_str, list):
            return [str(x).strip().strip('"\'') for x in author_str]
            
        # Handle string representation of lists
        if isinstance(author_str, str):
            # Remove brackets and split
            cleaned = author_str.strip('[]').strip()
            if not cleaned:
                return []
            # Split by comma and clean each ID
            return [x.strip().strip('"\'') for x in cleaned.split(',')]
            
        return []
    except Exception as e:
        print(f"Error parsing author string '{author_str}' of type {type(author_str)}: {e}")
        return []

def map_authors(author_ids, authors_map):
    try:
        if not author_ids:
            return []
        return [authors_map.get(str(aid)) for aid in author_ids if aid and str(aid) in authors_map]
    except Exception as e:
        print(f"Error mapping authors {author_ids}: {e}")
        return []

def add_author_names(df: pd.DataFrame, authors_map: Dict[str, str]) -> pd.DataFrame:
    """
    Convert author IDs to author names
    """
    print("Converting author IDs to names...")
    
    # Convert authors to list of IDs and then to names
    if 'authors' in df.columns:
        print("\nDebug: First few raw author entries and their types:")
        for entry in df['authors'].head():
            print(f"Value: {entry}, Type: {type(entry)}")
            
        df['authors'] = df['authors'].apply(parse_author_ids)
        print("\nDebug: First few parsed author IDs:", df['authors'].head().tolist())
        
        df['author_names'] = df['authors'].apply(lambda ids: map_authors(ids, authors_map))
        print("\nDebug: First few mapped author names:", df['author_names'].head().tolist())
        
        # Verify mapping success
        empty_names = df['author_names'].apply(lambda x: len(x) == 0).sum()
        print(f"\nStatistics:")
        print(f"Total records: {len(df)}")
        print(f"Records with empty author names: {empty_names}")
        print(f"Success rate: {((len(df) - empty_names) / len(df) * 100):.2f}%")
    else:
        print("Warning: 'authors' column not found in the dataset")
        df['author_names'] = [[]]
    
    return df

def add_genres(df: pd.DataFrame, genres_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Add genres to books
    """
    print("Adding genres to books...")
    
    def get_genres(book_id):
        return genres_map.get(str(book_id), [])
    
    df['genres'] = df['book_id'].apply(get_genres)
    return df

def transform_books(
    input_file: str = "filtered_books.parquet",
    output_file: str = "transformed_books.parquet",
    authors_file: str = "goodreads_book_authors.json",
    genres_file: str = "goodreads_book_genres_initial.json",
    add_authors: bool = True,
    add_genre: bool = True
) -> None:
    """
    Transform books data by adding author names and genres
    """
    print(f"Reading books from {input_file}...")
    df = pd.read_parquet(input_file)
    initial_columns = set(df.columns)
    
    if add_authors:
        authors_map = load_authors_mapping(authors_file)
        df = add_author_names(df, authors_map)
    
    if add_genre:
        genres_map = load_genres_mapping(genres_file)
        df = add_genres(df, genres_map)
    
    # Print summary of transformations
    print("\nTransformations summary:")
    new_columns = set(df.columns) - initial_columns
    if new_columns:
        print("Added columns:", ", ".join(new_columns))
    
    # Save transformed dataset
    df.to_parquet(output_file)
    print(f"\nSaved transformed dataset to: {output_file}")
    
    # Print sample of transformed books
    print("\nSample of transformed books:")
    sample = df.sample(min(5, len(df)))
    for _, book in sample.iterrows():
        print(f"\nTitle: {book['title']}")
        if 'author_names' in book:
            print(f"Authors: {', '.join(book['author_names'])}")
        if 'genres' in book:
            print(f"Genres: {', '.join(book['genres'])}")

def main():
    parser = argparse.ArgumentParser(description='Transform Goodreads books data with additional information')
    
    parser.add_argument('--input', type=str, default='filtered_books.parquet',
                      help='Input parquet file path (default: filtered_books.parquet)')
    
    parser.add_argument('--output', type=str, default='transformed_books.parquet',
                      help='Output parquet file path (default: transformed_books.parquet)')
    
    parser.add_argument('--authors-file', type=str, default='goodreads_book_authors.json',
                      help='Authors JSON file path (default: goodreads_book_authors.json)')
    
    parser.add_argument('--genres-file', type=str, default='goodreads_book_genres_initial.json',
                      help='Genres JSON file path (default: goodreads_book_genres_initial.json)')
    
    parser.add_argument('--skip-authors', action='store_true',
                      help='Skip adding author names')
    
    parser.add_argument('--skip-genres', action='store_true',
                      help='Skip adding genres')
    
    args = parser.parse_args()
    
    transform_books(
        input_file=args.input,
        output_file=args.output,
        authors_file=args.authors_file,
        genres_file=args.genres_file,
        add_authors=not args.skip_authors,
        add_genre=not args.skip_genres
    )

if __name__ == "__main__":
    main() 