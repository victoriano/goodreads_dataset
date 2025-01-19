import polars as pl
import argparse
from pathlib import Path
from typing import List, Dict, Literal, Optional
import openai
import os
from tqdm import tqdm
import json
from pydantic import BaseModel
import time
import asyncio
from typing import List
import aiohttp
from book_taxonomy import BOOK_TAXONOMY, CategoryType, get_category_type

class ShelfClassification(BaseModel):
    category: Literal["genre", "topic", "reading_status", "year_list", "book_format", "other"]
    confidence: float

class DetailedGenreClassification(BaseModel):
    category: str  # One of the categories from BOOK_TAXONOMY
    confidence: float
    is_fiction: bool

async def get_shelf_category(shelf_name: str, client: openai.AsyncOpenAI) -> ShelfClassification:
    """
    Use OpenAI to classify a shelf name into categories with confidence score
    """
    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at classifying Goodreads shelf names into categories.
You will be given a shelf name and should classify it into exactly one category with a confidence score.
Pay special attention to distinguish between:
- Genres: established categories of literature
- Topics: specific subjects or areas of interest
- Book Formats: physical format or medium of the book"""
                },
                {
                    "role": "user",
                    "content": f"""Classify this Goodreads shelf name: '{shelf_name}'
It must be classified as exactly ONE of these categories:
1. genre - if it represents an established category of literature (e.g., 'fantasy', 'mystery', 'romance', 'historical-fiction')
2. topic - if it represents a specific subject matter or area of interest (e.g., 'psychology', 'world-war-2', 'neuroscience', 'buddhism', 'climate-change')
3. reading_status - if it represents reading status (e.g., 'to-read', 'currently-reading', 'read')
4. year_list - if it represents a year-based list (e.g., '2024-books', 'read-in-2023')
5. book_format - if it represents the format or medium of the book (e.g., 'ebook', 'audiobook', 'hardcover', 'paperback', 'kindle', 'graphic-novel')
6. other - anything else that doesn't fit the above categories

Examples of classifications:
- 'civil-war' = topic (specific historical event)
- 'historical-fiction' = genre (type of literature)
- 'psychology' = topic (field of study)
- 'self-help' = genre (category of books)
- 'space' = topic (subject matter)
- 'science-fiction' = genre (type of literature)
- 'kindle-unlimited' = book_format (reading platform)
- 'audiobook' = book_format (listening format)
- 'manga' = book_format (comic format from Japan)
- 'hardback' = book_format (physical format)
- 'ebook' = book_format (digital format)

Provide your classification with a confidence score between 0 and 1."""
                }
            ],
            response_format=ShelfClassification
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error classifying shelf '{shelf_name}': {e}")
        return ShelfClassification(category="other", confidence=0.0)

async def get_detailed_genre(shelf_name: str, initial_category: str, client: openai.AsyncOpenAI) -> Optional[DetailedGenreClassification]:
    """
    Classify a shelf into one of the detailed genre categories from our taxonomy
    """
    try:
        # Only process genres and topics
        if initial_category not in ["genre", "topic"]:
            return None

        # Create a list of categories with their descriptions
        categories_list = []
        for cat, info in BOOK_TAXONOMY.items():
            desc = info["description"]
            # Add some common variations for better matching
            variations = []
            if cat == "fiction":
                variations = ["general fiction", "adult fiction"]
            elif cat == "childrens_literature":
                variations = ["children's", "kids", "children's books"]
            elif cat == "cultural_history":
                variations = ["culture", "cultural studies"]
            elif cat == "reference":
                variations = ["non-fiction", "nonfiction", "general non-fiction"]
            
            if variations:
                desc += f" (Also matches: {', '.join(variations)})"
            categories_list.append(f"- {cat}: {desc}")
        
        categories_str = "\n".join(categories_list)
        
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at classifying book shelves into detailed genre categories.
You will be given a shelf name and should classify it into exactly one category from our taxonomy.
The shelf has already been identified as either a genre or topic shelf.
You must choose one of the exact category names provided, even if it requires choosing a broader category."""
                },
                {
                    "role": "user",
                    "content": f"""Classify this Goodreads shelf name: '{shelf_name}'
This shelf was initially classified as: {initial_category}

Choose exactly ONE category from this taxonomy that best matches the shelf.
If you can't find an exact match, choose the most appropriate broader category:

{categories_str}

Provide your classification with a confidence score between 0 and 1.
The category MUST be one of the exact category names listed above (the part before the colon).
For example, for 'humor' books, use 'fiction' or 'literary_fiction' as they're the closest matches."""
                }
            ],
            response_format=DetailedGenreClassification
        )
        result = completion.choices[0].message.parsed
        # Add is_fiction based on the category type
        result.is_fiction = get_category_type(result.category) == "fiction"
        return result
    except Exception as e:
        print(f"Warning: Could not classify shelf '{shelf_name}' into detailed genre. Using broader category. Error: {e}")
        # Try to map to a broader category
        if initial_category == "genre":
            return DetailedGenreClassification(category="fiction", confidence=0.7, is_fiction=True)
        elif initial_category == "topic":
            return DetailedGenreClassification(category="reference", confidence=0.7, is_fiction=False)
        return None

async def classify_shelves_batch(shelves: List[dict], client: openai.AsyncOpenAI) -> List[dict]:
    """
    Classify a batch of shelves concurrently
    """
    # First classification tasks
    initial_tasks = []
    for shelf in shelves:
        task = get_shelf_category(shelf["popular_shelves"], client)
        initial_tasks.append(task)
    
    initial_results = await asyncio.gather(*initial_tasks)
    
    # Detailed genre classification tasks for genres and topics
    detailed_tasks = []
    detailed_indices = []  # Keep track of which shelves need detailed classification
    
    for i, (shelf, initial_result) in enumerate(zip(shelves, initial_results)):
        if initial_result.category in ["genre", "topic"]:
            task = get_detailed_genre(shelf["popular_shelves"], initial_result.category, client)
            detailed_tasks.append(task)
            detailed_indices.append(i)
    
    # Only gather detailed results if we have any tasks
    detailed_results_map = {}  # Map of index to result
    if detailed_tasks:
        detailed_results = await asyncio.gather(*detailed_tasks)
        for idx, result in zip(detailed_indices, detailed_results):
            detailed_results_map[idx] = result
    
    # Combine results
    classifications = []
    for i, (shelf, initial) in enumerate(zip(shelves, initial_results)):
        result = {
            "shelf_name": shelf["popular_shelves"],
            "count": shelf["count"],
            "category": initial.category,
            "confidence": initial.confidence,
        }
        
        # Add detailed classification if available
        if i in detailed_results_map and detailed_results_map[i] is not None:
            detailed = detailed_results_map[i]
            result.update({
                "detailed_category": detailed.category,
                "detailed_confidence": detailed.confidence,
                "is_fiction": detailed.is_fiction
            })
        
        classifications.append(result)
    
    return classifications

async def analyze_shelves_async(
    input_file: str = "goodreads_books.parquet",
    output_file: str = "popular_shelves.parquet",
    top_n: int = 1000,
    openai_api_key: str = None,
    batch_size: int = 1000
) -> None:
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
    
    client = openai.AsyncOpenAI(api_key=openai_api_key)
    
    # Process shelves in batches
    classifications = []
    shelves = shelf_counts.iter_rows(named=True)
    shelf_batches = list(shelves)
    
    with tqdm(total=len(shelf_batches)) as pbar:
        for i in range(0, len(shelf_batches), batch_size):
            batch = shelf_batches[i:i + batch_size]
            batch_results = await classify_shelves_batch(batch, client)
            classifications.extend(batch_results)
            pbar.update(len(batch))
    
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
    print(f"Average time per shelf: {classification_time/len(shelf_counts):.2f} seconds")
    print(f"Average shelves per second: {len(shelf_counts)/classification_time:.1f}")
    
    print("\nInitial Classification Summary:")
    initial_summary = (
        results_df.group_by("category")
        .agg(
            pl.len().alias("shelf_count"),
            pl.col("count").sum().alias("total_occurrences"),
            pl.col("confidence").mean().alias("avg_confidence")
        )
        .sort("total_occurrences", descending=True)
    )
    
    print("\nCategory distribution:")
    for row in initial_summary.iter_rows(named=True):
        print(f"{row['category']}:")
        print(f"  - Number of unique shelves: {row['shelf_count']}")
        print(f"  - Total occurrences: {row['total_occurrences']:,}")
        print(f"  - Average confidence: {row['avg_confidence']:.2f}")
    
    # Print detailed genre statistics for genres and topics
    detailed_df = results_df.filter(pl.col("detailed_category").is_not_null())
    if len(detailed_df) > 0:
        print("\nDetailed Genre Classification Summary:")
        detailed_summary = (
            detailed_df.group_by(["detailed_category", "is_fiction"])
            .agg(
                pl.len().alias("shelf_count"),
                pl.col("count").sum().alias("total_occurrences"),
                pl.col("detailed_confidence").mean().alias("avg_confidence")
            )
            .sort("total_occurrences", descending=True)
        )
        
        print("\nFiction categories:")
        fiction_summary = detailed_summary.filter(pl.col("is_fiction") == True)
        for row in fiction_summary.iter_rows(named=True):
            print(f"{row['detailed_category']}:")
            print(f"  - Number of unique shelves: {row['shelf_count']}")
            print(f"  - Total occurrences: {row['total_occurrences']:,}")
            print(f"  - Average confidence: {row['avg_confidence']:.2f}")
        
        print("\nNon-fiction categories:")
        non_fiction_summary = detailed_summary.filter(pl.col("is_fiction") == False)
        for row in non_fiction_summary.iter_rows(named=True):
            print(f"{row['detailed_category']}:")
            print(f"  - Number of unique shelves: {row['shelf_count']}")
            print(f"  - Total occurrences: {row['total_occurrences']:,}")
            print(f"  - Average confidence: {row['avg_confidence']:.2f}")
    
    # Print sample of most common shelves by category
    print("\nTop 5 shelves by initial category:")
    for category in ["genre", "topic", "reading_status", "year_list", "book_format"]:
        top_shelves = (
            results_df
            .filter(pl.col("category") == category)
            .sort("count", descending=True)
            .limit(5)
        )
        print(f"\n{category}:")
        for row in top_shelves.iter_rows(named=True):
            base_info = f"  - {row['shelf_name']}: {row['count']:,} occurrences (confidence: {row['confidence']:.2f})"
            if "detailed_category" in row and row["detailed_category"]:
                base_info += f"\n    Detailed: {row['detailed_category']} ({'fiction' if row['is_fiction'] else 'non-fiction'}, confidence: {row['detailed_confidence']:.2f})"
            print(base_info)

def analyze_shelves(
    input_file: str = "goodreads_books.parquet",
    output_file: str = "popular_shelves.parquet",
    top_n: int = 1000,
    openai_api_key: str = None,
    batch_size: int = 500
) -> None:
    """Wrapper function to run async analysis"""
    asyncio.run(analyze_shelves_async(
        input_file=input_file,
        output_file=output_file,
        top_n=top_n,
        openai_api_key=openai_api_key,
        batch_size=batch_size
    ))

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
    
    parser.add_argument('--batch-size', type=int, default=500,
                      help='Number of shelves to process in parallel (default: 500). With tier 5 access, this can go up to 30000.')
    
    args = parser.parse_args()
    
    if args.classify:
        analyze_shelves(
            input_file=args.input,
            output_file=args.output,
            top_n=args.top_n,
            openai_api_key=args.openai_api_key,
            batch_size=args.batch_size
        )
    else:
        aggregate_shelves(
            input_file=args.input,
            output_file=args.output,
            top_n=args.top_n
        )

if __name__ == "__main__":
    main() 