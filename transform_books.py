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

def load_genres_mapping(classified_shelves_file: str = "classified_shelves_10K.parquet", 
                        input_df: pd.DataFrame = None) -> Dict[str, Dict[str, Any]]:
    """
    Load classified_shelves_10K.parquet as a dictionary to classify the original popular_shelves
    from the input DataFrame into curated categories.

    Returns a dictionary mapping book_id (as string) to a dictionary containing:
    - curated_shelves: list of shelf names that are either "genre" or "topic"
    - shelf_categories: list of the shelf's high-level categories ("genre", "topic", etc.)
    - detail_categories: list of the shelf's detailed categories
    - is_fiction_ratio: percentage of shelves classified as fiction (0 <= ratio <= 1)
    """
    print(f"Loading classified shelves from {classified_shelves_file}...")
    try:
        shelves_df = pd.read_parquet(classified_shelves_file)
    except Exception as e:
        print(f"Error reading classified_shelves_file ({classified_shelves_file}): {e}")
        return {}

    print(f"classified_shelves DataFrame shape: {shelves_df.shape}")
    print("First few rows of shelves_df:")
    print(shelves_df.head(5))

    # Build a lookup for shelf classification details
    shelf_classification = {}
    loaded_count = 0
    for _, row in shelves_df.iterrows():
        shelf_name = row.get('shelf_name')
        category = row.get('category')
        if pd.isna(category) or not shelf_name:
            continue

        # Handle missing or NaN fields
        detailed_category = row.get('detailed_category')
        if pd.isna(detailed_category):
            detailed_category = None
        is_fiction = row.get('is_fiction')
        if pd.isna(is_fiction):
            is_fiction = None

        shelf_classification[shelf_name] = {
            'category': category,
            'detailed_category': detailed_category,
            'is_fiction': is_fiction
        }
        loaded_count += 1

    print(f"Loaded classification details for {loaded_count:,} shelves")

    def process_book_shelves(row: pd.Series) -> Dict[str, Any]:
        if 'book_id' not in row:
            print(f"Warning: 'book_id' missing in row: {row}")
            return {}

        book_id = str(row['book_id'])

        # Retrieve the original popular_shelves from the row
        try:
            popular_shelves_data = row['popular_shelves']
        except KeyError:
            print(f"Warning: 'popular_shelves' missing for book_id={book_id}")
            popular_shelves_data = []

        # Convert popular_shelves into a usable Python list
        if isinstance(popular_shelves_data, np.ndarray):
            # Convert numpy array to list
            popular_shelves = popular_shelves_data.tolist()
        elif isinstance(popular_shelves_data, list):
            # Already a list; just assign
            popular_shelves = popular_shelves_data
        elif isinstance(popular_shelves_data, str):
            import ast
            # Attempt to parse the string representation of a list
            try:
                popular_shelves = ast.literal_eval(popular_shelves_data)
            except Exception as e:
                print(f"Error parsing popular_shelves string for book_id={book_id}: {e}")
                popular_shelves = []
        else:
            print(
                f"Unexpected popular_shelves type for book_id={book_id}. "
                f"Got: {type(popular_shelves_data)}"
            )
            popular_shelves = []

        # Prepare classification data structure
        book_data = {
            'popular_shelves': popular_shelves,
            'curated_shelves': [],
            'shelf_categories': [],
            'detail_categories': [],
            'fiction_count': 0,
            'total_classified': 0
        }

        # Classify each shelf
        for shelf_name in popular_shelves:
            shelf_info = shelf_classification.get(shelf_name)
            if not shelf_info:
                # Shelf not found in classification file
                continue

            # Include the shelf if it is a "genre" or "topic"
            if shelf_info['category'] in ['genre', 'topic']:
                if shelf_name not in book_data['curated_shelves']:
                    book_data['curated_shelves'].append(shelf_name)
                if shelf_info['category'] not in book_data['shelf_categories']:
                    book_data['shelf_categories'].append(shelf_info['category'])
                if (
                    shelf_info['detailed_category'] 
                    and shelf_info['detailed_category'] not in book_data['detail_categories']
                ):
                    book_data['detail_categories'].append(shelf_info['detailed_category'])

            # Count fiction vs. non-fiction
            if shelf_info['is_fiction'] is not None:
                book_data['total_classified'] += 1
                if shelf_info['is_fiction'] is True:
                    book_data['fiction_count'] += 1

        # Calculate the ratio of shelves classified as fiction
        total = book_data['total_classified']
        book_data['is_fiction_ratio'] = book_data['fiction_count'] / total if total else None

        # Remove temporary fields
        del book_data['fiction_count']
        del book_data['total_classified']
        return book_data

    genres_map = {}
    if input_df is not None:
        print(f"\nProcessing input_df: {input_df.shape[0]:,} rows, columns: {list(input_df.columns)}")
        print("\nSample of input_df (first 5 rows):")
        print(input_df.head(5))

        # Creating a map for each book
        for _, row in input_df.iterrows():
            classification_result = process_book_shelves(row)
            if classification_result:
                book_id_str = str(row['book_id'])
                genres_map[book_id_str] = classification_result

    print(f"\nCreated classification mapping for {len(genres_map):,} books "
          f"out of {0 if input_df is None else len(input_df):,} total input records.")
    return genres_map

def parse_author_ids(author_str):
    """Convert any author string/array format to a list of string IDs"""
    try:
        # Handle numpy arrays first
        if isinstance(author_str, np.ndarray):
            return author_str.tolist()  # Convert directly to list
        
        # Then check for NA values (now safe to do so)
        if pd.isna(author_str):
            return []
        
        # Handle string representation
        if isinstance(author_str, str):
            import ast
            try:
                # Safely evaluate the string as a literal (e.g., "[1, 2]")
                return ast.literal_eval(author_str)
            except:
                # If that fails, try manual parsing
                cleaned = author_str.strip('[]').strip()
                if not cleaned:
                    return []
                return [x.strip().strip('"\'') for x in cleaned.split(',')]
        
        # Handle plain Python lists
        if isinstance(author_str, list):
            return author_str
        
        # Fallback
        return []
    except Exception as e:
        print(f"Error parsing author string '{author_str}' of type {type(author_str)}: {e}")
        return []

def map_authors(author_ids: List[str], authors_map: Dict[str, str]) -> List[str]:
    """Map author IDs to their names"""
    try:
        if not author_ids:
            return []
        
        # Convert all IDs to strings and look up in map
        return [authors_map[str(aid)] for aid in author_ids if str(aid) in authors_map]
    except Exception as e:
        print(f"Error mapping authors {author_ids}: {e}")
        return []

def add_author_names(df: pd.DataFrame, authors_map: Dict[str, str]) -> pd.DataFrame:
    """Convert author IDs to author names"""
    print("Converting author IDs to names...")
    
    if 'authors' in df.columns:
        print("\nDebug: First few raw author entries and their types:")
        for entry in df['authors'].head():
            print(f"Value: {entry}, Type: {type(entry)}")
        
        # Parse author IDs
        df['authors'] = df['authors'].apply(parse_author_ids)
        print("\nDebug: First few parsed author IDs:", df['authors'].head().tolist())
        
        # Map to author names
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

def add_genres(df: pd.DataFrame, genres_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Add genre and shelf classification information to books
    """
    print("Adding shelf classification to books...")
    
    def get_book_classification(book_id):
        book_data = genres_map.get(str(book_id), {
            'popular_shelves': [],
            'curated_shelves': [],
            'shelf_categories': [],
            'detail_categories': [],
            'is_fiction_ratio': None
        })
        return book_data
    
    # Add new columns for each classification aspect
    classifications = df['book_id'].apply(get_book_classification)
    df['popular_shelves'] = classifications.apply(lambda x: x['popular_shelves'])
    df['curated_shelves'] = classifications.apply(lambda x: x['curated_shelves'])
    df['shelf_categories'] = classifications.apply(lambda x: x['shelf_categories'])
    df['detail_categories'] = classifications.apply(lambda x: x['detail_categories'])
    df['is_fiction_ratio'] = classifications.apply(lambda x: x['is_fiction_ratio'])
    
    # Print some statistics about the classification
    total_books = len(df)
    books_with_shelves = df['curated_shelves'].apply(len).gt(0).sum()
    print(f"\nClassification Statistics:")
    print(f"Total books: {total_books:,}")
    print(f"Books with classified shelves: {books_with_shelves:,}")
    print(f"Success rate: {(books_with_shelves / total_books * 100):.2f}%")
    
    return df

def load_books_mapping(books_file: str = "goodreads_books.parquet") -> Dict[str, str]:
    """
    Load books data and create a mapping from book_id to book title
    """
    print(f"Loading books from {books_file}...")
    df = pd.read_parquet(books_file)
    books_map = df.set_index('book_id')['title'].to_dict()
    print(f"Loaded {len(books_map):,} book titles")
    return books_map

def parse_similar_book_ids(similar_str):
    """Convert similar books string/array format to a list of string IDs"""
    try:
        # Handle numpy arrays first
        if isinstance(similar_str, np.ndarray):
            return [str(x) for x in similar_str.tolist()]  # Convert directly to list
        
        # Then check for NA values (now safe to do so)
        if pd.isna(similar_str):
            return []
        
        # Handle string representation
        if isinstance(similar_str, str):
            import ast
            try:
                # Safely evaluate the string as a literal (e.g., "[1, 2]")
                return [str(x) for x in ast.literal_eval(similar_str)]
            except:
                # If that fails, try manual parsing
                cleaned = similar_str.strip('[]').strip()
                if not cleaned:
                    return []
                return [x.strip().strip('"\'') for x in cleaned.split(',')]
        
        # Handle plain Python lists
        if isinstance(similar_str, list):
            return [str(x) for x in similar_str]
        
        # Fallback
        return []
    except Exception as e:
        print(f"Error parsing similar books string '{similar_str}' of type {type(similar_str)}: {e}")
        return []

def add_similar_book_titles(df: pd.DataFrame, books_map: Dict[str, str]) -> pd.DataFrame:
    """Convert similar book IDs to book titles"""
    print("Converting similar book IDs to titles...")
    
    if 'similar_books' in df.columns:
        # Parse similar book IDs
        df['similar_books'] = df['similar_books'].apply(parse_similar_book_ids)
        
        # Map to book titles
        df['similar_book_titles'] = df['similar_books'].apply(
            lambda ids: [books_map.get(bid, "Unknown") for bid in ids if bid in books_map]
        )
        
        # Verify mapping success
        empty_similar = df['similar_book_titles'].apply(lambda x: len(x) == 0).sum()
        print(f"\nSimilar Books Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Records with no similar books: {empty_similar}")
        print(f"Success rate: {((len(df) - empty_similar) / len(df) * 100):.2f}%")
    else:
        print("Warning: 'similar_books' column not found in the dataset")
        df['similar_book_titles'] = [[]]
    
    return df

def add_publication_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a publication_date field in dd/mm/yyyy format using day, month and publication_year
    """
    print("Creating publication date field...")
    
    def format_date(row):
        try:
            # Default to 1 if day or month is missing
            day = 1 if pd.isna(row['publication_day']) else int(row['publication_day'])
            month = 1 if pd.isna(row['publication_month']) else int(row['publication_month'])
            year = int(row['publication_year']) if not pd.isna(row['publication_year']) else None
            
            # Return None if year is missing
            if year is None:
                return None
            
            # Validate day and month values
            day = min(max(1, day), 31)  # Keep day between 1 and 31
            month = min(max(1, month), 12)  # Keep month between 1 and 12
            
            return f"{day:02d}/{month:02d}/{year}"
        except (ValueError, TypeError):
            return None
    
    df['publication_date'] = df.apply(format_date, axis=1)
    
    # Print some statistics
    total_dates = len(df)
    valid_dates = df['publication_date'].notna().sum()
    print(f"Created {valid_dates:,} valid dates out of {total_dates:,} records")
    print(f"Success rate: {(valid_dates / total_dates * 100):.2f}%")
    
    return df

def select_and_reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and reorder columns in the specified order.
    Missing columns will be filled with None/empty values.
    Then sort by ratings_count in descending order.
    """
    desired_columns = [
        'title', 'description', 'author_names', 'popular_shelves', 'curated_shelves', 'shelf_categories', 
        'detail_categories', 'is_fiction_ratio', 'similar_book_titles',
        'publication_date', 'publication_year', 'url', 'num_pages', 'ratings_count', 'average_rating',
        'text_reviews_count', 'publisher', 'language_code', 'country_code',
        'format', 'publication_month', 'publication_day', 'is_ebook',
        'edition_information', 'image_url', 'isbn', 'kindle_asin', 'book_id'
    ]
    
    # Create missing columns with appropriate empty values
    for col in desired_columns:
        if col not in df.columns:
            if col in ['author_names', 'popular_shelves', 'curated_shelves', 'shelf_categories', 'detail_categories', 'similar_book_titles']:
                df[col] = [[]]  # Empty list for list columns
            elif col == 'is_fiction_ratio':
                df[col] = None  # None for the ratio
            else:
                df[col] = None  # None for scalar columns
    
    # Select and reorder columns
    df = df[desired_columns]
    
    # Sort by ratings_count in descending order
    print("Sorting books by ratings count...")
    df = df.sort_values('ratings_count', ascending=False)
    
    return df

def transform_books(
    input_file: str = "filtered_books.parquet",
    output_file: str = "transformed_books.parquet",
    authors_file: str = "goodreads_book_authors.json",
    genres_file: str = "classified_shelves_10K.parquet",
    book_shelves_file: str = "goodreads_book_shelves.json",
    books_file: str = "goodreads_books.parquet",
    add_authors: bool = True,
    add_genre: bool = True,
    add_similar: bool = True
) -> None:
    """
    Transform books data by adding author names, genres, and similar book titles
    """
    print(f"Reading books from {input_file}...")
    df = pd.read_parquet(input_file)
    initial_columns = set(df.columns)
    
    if add_authors:
        authors_map = load_authors_mapping(authors_file)
        df = add_author_names(df, authors_map)
    
    if add_genre:
        genres_map = load_genres_mapping(genres_file, df)
        df = add_genres(df, genres_map)
    
    if add_similar:
        books_map = load_books_mapping(books_file)
        df = add_similar_book_titles(df, books_map)
    
    # Add publication date
    df = add_publication_date(df)
    
    # Print summary of transformations
    print("\nTransformations summary:")
    new_columns = set(df.columns) - initial_columns
    if new_columns:
        print("Added columns:", ", ".join(new_columns))
    
    # Select and reorder columns
    print("\nSelecting and reordering columns...")
    df = select_and_reorder_columns(df)
    
    # Save transformed dataset
    df.to_parquet(output_file)
    print(f"\nSaved transformed dataset to: {output_file}")
    
    # Print sample of transformed books
    print("\nSample of transformed books:")
    sample = df.sample(min(5, len(df)))
    for _, book in sample.iterrows():
        print(f"\nTitle: {book['title']}")
        if book['author_names'] and len(book['author_names']) > 0:
            print(f"Authors: {', '.join(book['author_names'])}")
        if book['popular_shelves'] and len(book['popular_shelves']) > 0:
            print(f"Popular Shelves: {', '.join(book['popular_shelves'][:5])}")  # Show first 5 shelves
        if book['curated_shelves'] and len(book['curated_shelves']) > 0:
            print(f"Curated Shelves: {', '.join(book['curated_shelves'])}")
        if book['shelf_categories'] and len(book['shelf_categories']) > 0:
            print(f"Categories: {', '.join(book['shelf_categories'])}")
        if book['detail_categories'] and len(book['detail_categories']) > 0:
            print(f"Detailed Categories: {', '.join(book['detail_categories'])}")
        if book['is_fiction_ratio'] is not None:
            print(f"Fiction Ratio: {book['is_fiction_ratio']:.2f}")
        if book['similar_book_titles'] and len(book['similar_book_titles']) > 0:
            print(f"Similar Books: {', '.join(book['similar_book_titles'][:5])}")  # Show first 5 similar books
        if book['publication_date']:
            print(f"Publication Date: {book['publication_date']}")

def main():
    parser = argparse.ArgumentParser(description='Transform Goodreads books data with additional information')
    
    parser.add_argument('--input', type=str, default='filtered_books.parquet',
                      help='Input parquet file path (default: filtered_books.parquet)')
    
    parser.add_argument('--output', type=str, default='transformed_books.parquet',
                      help='Output parquet file path (default: transformed_books.parquet)')
    
    parser.add_argument('--authors-file', type=str, default='goodreads_book_authors.json',
                      help='Authors JSON file path (default: goodreads_book_authors.json)')
    
    parser.add_argument('--genres-file', type=str, default='classified_shelves_10K.parquet',
                      help='Classified shelves parquet file path (default: classified_shelves_10K.parquet)')
    
    parser.add_argument('--book-shelves-file', type=str, default='goodreads_book_shelves.json',
                      help='Book shelves JSON file path (default: goodreads_book_shelves.json)')
    
    parser.add_argument('--books-file', type=str, default='goodreads_books.parquet',
                      help='Books parquet file path (default: goodreads_books.parquet)')
    
    parser.add_argument('--skip-authors', action='store_true',
                      help='Skip adding author names')
    
    parser.add_argument('--skip-genres', action='store_true',
                      help='Skip adding genres')
    
    parser.add_argument('--skip-similar', action='store_true',
                      help='Skip adding similar book titles')
    
    args = parser.parse_args()
    
    transform_books(
        input_file=args.input,
        output_file=args.output,
        authors_file=args.authors_file,
        genres_file=args.genres_file,
        book_shelves_file=args.book_shelves_file,
        books_file=args.books_file,
        add_authors=not args.skip_authors,
        add_genre=not args.skip_genres,
        add_similar=not args.skip_similar
    )

if __name__ == "__main__":
    main() 