import pandas as pd
import argparse
from pathlib import Path

def filter_books(
    input_file: str = "goodreads_books.parquet",
    output_file: str = "filtered_books.parquet",
    min_reviews: int = 1000,
    min_rating: float = None,
    min_avg_rating: float = None
) -> None:
    """
    Filter books based on various criteria
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output filtered parquet file
        min_reviews: Minimum number of text reviews (default: 1000)
        min_rating: Minimum number of ratings
        min_avg_rating: Minimum average rating
    """
    print(f"Reading parquet file: {input_file}")
    df = pd.read_parquet(input_file)
    
    initial_count = len(df)
    print(f"\nInitial number of books: {initial_count:,}")
    
    # Convert columns to numeric if they aren't already
    df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
    df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
    
    # Apply filters
    filters = []
    
    if min_reviews:
        filters.append(f'text_reviews_count >= {min_reviews}')
        
    if min_rating:
        filters.append(f'ratings_count >= {min_rating}')
        
    if min_avg_rating:
        filters.append(f'average_rating >= {min_avg_rating}')
    
    # Apply all filters
    filter_expr = ' & '.join(filters)
    filtered_df = df.query(filter_expr)
    
    # Print statistics
    print("\nFilter criteria:")
    if min_reviews:
        print(f"- Minimum reviews: {min_reviews:,}")
    if min_rating:
        print(f"- Minimum ratings: {min_rating:,}")
    if min_avg_rating:
        print(f"- Minimum average rating: {min_avg_rating}")
    
    print(f"\nFiltered number of books: {len(filtered_df):,}")
    print(f"Removed {initial_count - len(filtered_df):,} books")
    
    # Save filtered dataset
    filtered_df.to_parquet(output_file)
    print(f"\nSaved filtered dataset to: {output_file}")
    
    # Print sample of filtered books
    print("\nSample of filtered books:")
    sample = filtered_df.sample(min(5, len(filtered_df)))
    for _, book in sample.iterrows():
        print(f"\nTitle: {book['title']}")
        print(f"Reviews: {int(book['text_reviews_count']):,}")
        print(f"Ratings: {int(book['ratings_count']):,}")
        print(f"Avg Rating: {float(book['average_rating']):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Filter Goodreads books based on various criteria')
    
    parser.add_argument('--input', type=str, default='goodreads_books.parquet',
                      help='Input parquet file path (default: goodreads_books.parquet)')
    
    parser.add_argument('--output', type=str, default='filtered_books.parquet',
                      help='Output parquet file path (default: filtered_books.parquet)')
    
    parser.add_argument('--min-reviews', type=int, default=1000,
                      help='Minimum number of text reviews (default: 1000)')
    
    parser.add_argument('--min-ratings', type=int,
                      help='Minimum number of ratings (optional)')
    
    parser.add_argument('--min-avg-rating', type=float,
                      help='Minimum average rating (optional)')
    
    args = parser.parse_args()
    
    filter_books(
        input_file=args.input,
        output_file=args.output,
        min_reviews=args.min_reviews,
        min_rating=args.min_ratings,
        min_avg_rating=args.min_avg_rating
    )

if __name__ == "__main__":
    main() 