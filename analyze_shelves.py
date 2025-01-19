import polars as pl
import argparse
from pathlib import Path
from typing import List, Dict, Literal
import openai
import os
from tqdm import tqdm
import json
from pydantic import BaseModel
import time

class ShelfClassification(BaseModel):
    category: Literal["genre", "reading_status", "year_list", "other"]
    confidence: float

def get_shelf_category(shelf_name: str, client: openai.OpenAI) -> ShelfClassification:
    """
    Use OpenAI to classify a shelf name into one of three categories with confidence score
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at classifying Goodreads shelf names into categories.
You will be given a shelf name and should classify it into exactly one category with a confidence score."""
                },
                {
                    "role": "user",
                    "content": f"""Classify this Goodreads shelf name: '{shelf_name}'
It must be classified as exactly ONE of these categories:
1. genre - if it represents a book genre (e.g., 'fantasy', 'mystery', 'romance')
2. reading_status - if it represents reading status (e.g., 'to-read', 'currently-reading', 'read')
3. year_list - if it represents a year-based list (e.g., '2024-books', 'read-in-2023')
4. other - anything else

Provide your classification with a confidence score between 0 and 1."""
                }
            ],
            response_format=ShelfClassification
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error classifying shelf '{shelf_name}': {e}")
        return ShelfClassification(category="other", confidence=0.0)

def analyze_shelves(
    input_file: str = "goodreads_books.parquet",
    output_file: str = "popular_shelves.parquet",
    top_n: int = 1000,
    openai_api_key: str = None
) -> None:
    """
    Analyze popular shelves from Goodreads books dataset
    
    Args:
        input_file: Path to input parquet file with books data
        output_file: Path to output parquet file for shelf analysis
        top_n: Number of top shelves to analyze (default: 1000)
        openai_api_key: OpenAI API key for classification
    """
    start_time = time.time()
    print(f"Reading parquet file: {input_file}")
    
    # Read the parquet file using Polars
    df = pl.scan_parquet(input_file)
    
    # Explode the popular_shelves array and group by shelf name
    shelf_counts = (
        df.select(pl.col("popular_shelves"))
        .explode("popular_shelves")
        .group_by("popular_shelves")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(top_n)
        .collect()
    )
    
    aggregation_time = time.time() - start_time
    print(f"\nAggregation completed in {aggregation_time:.2f} seconds")
    
    print(f"\nAnalyzing top {top_n} shelves...")
    classification_start = time.time()
    
    # Initialize OpenAI client
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Classify each shelf
    classifications = []
    for shelf in tqdm(shelf_counts.iter_rows(named=True), total=len(shelf_counts)):
        shelf_name = shelf["popular_shelves"]
        result = get_shelf_category(shelf_name, client)
        classifications.append({
            "shelf_name": shelf_name,
            "count": shelf["count"],
            "category": result.category,
            "confidence": result.confidence
        })
    
    # Convert to Polars DataFrame and save
    results_df = pl.DataFrame(classifications)
    results_df.write_parquet(output_file)
    
    classification_time = time.time() - classification_start
    total_time = time.time() - start_time
    
    # Print summary statistics
    print("\nTiming Summary:")
    print(f"Aggregation time: {aggregation_time:.2f} seconds")
    print(f"Classification time: {classification_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per shelf classification: {classification_time/len(shelf_counts):.2f} seconds")
    
    print("\nClassification Summary:")
    summary = (
        results_df.group_by("category")
        .agg(
            pl.count().alias("shelf_count"),
            pl.col("count").sum().alias("total_occurrences"),
            pl.col("confidence").mean().alias("avg_confidence")
        )
        .sort("total_occurrences", descending=True)
    )
    
    print("\nCategory distribution:")
    for row in summary.iter_rows(named=True):
        print(f"{row['category']}:")
        print(f"  - Number of unique shelves: {row['shelf_count']}")
        print(f"  - Total occurrences: {row['total_occurrences']:,}")
        print(f"  - Average confidence: {row['avg_confidence']:.2f}")
    
    # Print sample of most common shelves by category
    print("\nTop 5 shelves by category:")
    for category in ["genre", "reading_status", "year_list"]:
        top_shelves = (
            results_df
            .filter(pl.col("category") == category)
            .sort("count", descending=True)
            .limit(5)
        )
        print(f"\n{category}:")
        for row in top_shelves.iter_rows(named=True):
            print(f"  - {row['shelf_name']}: {row['count']:,} occurrences (confidence: {row['confidence']:.2f})")

def aggregate_shelves(
    input_file: str = "goodreads_books.parquet",
    output_file: str = "popular_shelves.parquet",
    top_n: int = 1000
) -> None:
    """
    Aggregate and count popular shelves from Goodreads books dataset without AI classification
    
    Args:
        input_file: Path to input parquet file with books data
        output_file: Path to output parquet file for shelf analysis
        top_n: Number of top shelves to analyze (default: 1000)
    """
    start_time = time.time()
    print(f"Reading parquet file: {input_file}")
    
    # Read the parquet file using Polars
    df = pl.scan_parquet(input_file)
    
    # Explode the popular_shelves array and group by shelf name
    shelf_counts = (
        df.select(pl.col("popular_shelves"))
        .explode("popular_shelves")
        .group_by("popular_shelves")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(top_n)
        .collect()
    )
    
    # Save results
    shelf_counts.write_parquet(output_file)
    
    total_time = time.time() - start_time
    
    # Print summary statistics
    total_occurrences = shelf_counts["count"].sum()
    print("\nTiming Summary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    print("\nSummary Statistics:")
    print(f"Total unique shelves analyzed: {len(shelf_counts)}")
    print(f"Total shelf occurrences: {total_occurrences:,}")
    
    # Print top shelves
    print("\nTop 20 most common shelves:")
    for row in shelf_counts.head(20).iter_rows(named=True):
        print(f"  - {row['popular_shelves']}: {row['count']:,} occurrences ({row['count']/total_occurrences*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze Goodreads popular shelves')
    
    parser.add_argument('--input', type=str, default='goodreads_books.parquet',
                      help='Input parquet file path (default: goodreads_books.parquet)')
    
    parser.add_argument('--output', type=str, default='popular_shelves.parquet',
                      help='Output parquet file path (default: popular_shelves.parquet)')
    
    parser.add_argument('--top-n', type=int, default=1000,
                      help='Number of top shelves to analyze (default: 1000)')
    
    parser.add_argument('--classify', action='store_true',
                      help='Use AI to classify shelves into categories')
    
    parser.add_argument('--openai-api-key', type=str,
                      help='OpenAI API key (required only with --classify)')
    
    args = parser.parse_args()
    
    if args.classify:
        analyze_shelves(
            input_file=args.input,
            output_file=args.output,
            top_n=args.top_n,
            openai_api_key=args.openai_api_key
        )
    else:
        aggregate_shelves(
            input_file=args.input,
            output_file=args.output,
            top_n=args.top_n
        )

if __name__ == "__main__":
    main() 