#!/usr/bin/env python3
"""
Dataset Audit Script for ML Challenge 2026
Performs comprehensive inspection of all TSV files using chunked reading
"""

import os
import pandas as pd
from collections import defaultdict, Counter
import sys

# Configuration
CHUNK_SIZE = 10000
SAMPLE_SIZE = 5

def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)

def inspect_tsv_file(filepath, file_type="source"):
    """
    Inspect a TSV file using chunked reading
    Returns comprehensive statistics about the file
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'='*80}")

    stats = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'file_size_mb': get_file_size(filepath),
        'total_rows': 0,
        'columns': None,
        'column_order': None,
        'missing_counts': defaultdict(int),
        'empty_counts': defaultdict(int),
        'unique_entity_ids': set(),
        'country_dist': Counter(),
        'entity_id_prefixes': Counter(),
        'sample_rows': [],
        'malformed_rows': [],
        'duplicate_entity_ids': [],
        'parsing_errors': []
    }

    try:
        # First, read just the header
        with open(filepath, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            stats['columns'] = header_line.split('\t')
            stats['column_order'] = stats['columns']

        print(f"File size: {stats['file_size_mb']:.2f} MB")
        print(f"Columns: {stats['columns']}")

        # Track entity_id duplicates
        seen_entity_ids = set()

        # Read in chunks
        chunk_num = 0
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
            chunk_num += 1

            # Count rows
            stats['total_rows'] += len(chunk)

            # Collect sample rows (from first chunk only)
            if chunk_num == 1 and len(stats['sample_rows']) < SAMPLE_SIZE:
                for idx in range(min(SAMPLE_SIZE, len(chunk))):
                    stats['sample_rows'].append(chunk.iloc[idx].to_dict())

            # Check for missing/empty values
            for col in chunk.columns:
                missing = chunk[col].isna().sum()
                empty = (chunk[col] == '').sum()
                stats['missing_counts'][col] += missing
                stats['empty_counts'][col] += empty

            # Analyze entity_id column
            if 'entity_id' in chunk.columns:
                entity_ids = chunk['entity_id'].dropna()

                # Check for duplicates
                for eid in entity_ids:
                    if eid in seen_entity_ids:
                        stats['duplicate_entity_ids'].append(eid)
                    else:
                        seen_entity_ids.add(eid)
                        stats['unique_entity_ids'].add(eid)

                # Extract prefixes
                for eid in entity_ids:
                    if pd.notna(eid) and '-' in str(eid):
                        prefix = str(eid).split('-')[0]
                        stats['entity_id_prefixes'][prefix] += 1

            # Analyze country distribution
            if 'country' in chunk.columns:
                countries = chunk['country'].dropna()
                stats['country_dist'].update(countries)

            # Check for malformed rows (rows with wrong number of columns)
            expected_cols = len(stats['columns'])

            if chunk_num % 10 == 0:
                print(f"  Processed {stats['total_rows']:,} rows...", end='\r')

        print(f"  Processed {stats['total_rows']:,} rows - COMPLETE")

        # Convert sets to counts for reporting
        stats['unique_entity_id_count'] = len(stats['unique_entity_ids'])
        stats['unique_entity_ids'] = None  # Clear to save memory

    except Exception as e:
        stats['parsing_errors'].append(str(e))
        print(f"ERROR: {e}")

    return stats

def inspect_ground_truth(filepath):
    """
    Inspect the ground truth file
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'='*80}")

    stats = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'file_size_mb': get_file_size(filepath),
        'total_rows': 0,
        'columns': None,
        'column_order': None,
        'sample_rows': [],
        'empty_matches': 0,
        'match_distribution': Counter(),
        'source1_ids': set(),
        'matched_s2_ids': set(),
        'matched_s3_ids': set()
    }

    try:
        # Read header
        with open(filepath, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            stats['columns'] = header_line.split('\t')
            stats['column_order'] = stats['columns']

        print(f"File size: {stats['file_size_mb']:.2f} MB")
        print(f"Columns: {stats['columns']}")

        # Read in chunks
        chunk_num = 0
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
            chunk_num += 1
            stats['total_rows'] += len(chunk)

            # Collect sample rows
            if chunk_num == 1 and len(stats['sample_rows']) < SAMPLE_SIZE:
                for idx in range(min(SAMPLE_SIZE, len(chunk))):
                    stats['sample_rows'].append(chunk.iloc[idx].to_dict())

            # Analyze matches
            for _, row in chunk.iterrows():
                s1_id = row.get('source1_entity_id', '')
                matched_ids = row.get('matched_entity_ids', '')

                if pd.notna(s1_id):
                    stats['source1_ids'].add(s1_id)

                if pd.isna(matched_ids) or matched_ids == '':
                    stats['empty_matches'] += 1
                else:
                    # Parse matched IDs
                    match_list = [m.strip() for m in str(matched_ids).split(',') if m.strip()]
                    stats['match_distribution'][len(match_list)] += 1

                    for mid in match_list:
                        if mid.startswith('S2-'):
                            stats['matched_s2_ids'].add(mid)
                        elif mid.startswith('S3-'):
                            stats['matched_s3_ids'].add(mid)

            if chunk_num % 10 == 0:
                print(f"  Processed {stats['total_rows']:,} rows...", end='\r')

        print(f"  Processed {stats['total_rows']:,} rows - COMPLETE")

        # Convert to counts
        stats['unique_source1_ids'] = len(stats['source1_ids'])
        stats['unique_matched_s2_ids'] = len(stats['matched_s2_ids'])
        stats['unique_matched_s3_ids'] = len(stats['matched_s3_ids'])

    except Exception as e:
        print(f"ERROR: {e}")
        stats['parsing_errors'] = [str(e)]

    return stats

def print_report(all_stats):
    """
    Generate comprehensive report
    """
    print("\n\n")
    print("="*100)
    print(" DATASET AUDIT REPORT ".center(100, "="))
    print("="*100)

    # 1. FILE INVENTORY
    print("\n[1] FILE INVENTORY")
    print("-" * 100)
    for stats in all_stats:
        print(f"  {stats['filename']:<30} | Size: {stats['file_size_mb']:>8.2f} MB | Rows: {stats['total_rows']:>10,}")

    # 2. SCHEMA
    print("\n[2] SCHEMA")
    print("-" * 100)
    for stats in all_stats:
        print(f"\n  {stats['filename']}:")
        print(f"    Columns: {stats['column_order']}")
        print(f"    Column count: {len(stats['column_order'])}")

    # 3. DATASET SIZES
    print("\n[3] DATASET SIZES")
    print("-" * 100)
    total_size = sum(s['file_size_mb'] for s in all_stats)
    total_rows = sum(s['total_rows'] for s in all_stats)
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Total rows: {total_rows:,}")
    print(f"\n  Breakdown:")
    for stats in all_stats:
        print(f"    {stats['filename']:<30} | {stats['total_rows']:>10,} rows")

    # 4. MISSING VALUES
    print("\n[4] MISSING VALUES")
    print("-" * 100)
    for stats in all_stats:
        if 'missing_counts' in stats and stats['missing_counts']:
            print(f"\n  {stats['filename']}:")
            total = stats['total_rows']
            for col, count in stats['missing_counts'].items():
                pct = (count / total * 100) if total > 0 else 0
                empty = stats['empty_counts'].get(col, 0)
                empty_pct = (empty / total * 100) if total > 0 else 0
                print(f"    {col:<25} | Missing: {count:>8,} ({pct:>5.2f}%) | Empty: {empty:>8,} ({empty_pct:>5.2f}%)")

    # 5. COUNTRY DISTRIBUTION
    print("\n[5] COUNTRY DISTRIBUTION")
    print("-" * 100)
    for stats in all_stats:
        if 'country_dist' in stats and stats['country_dist']:
            print(f"\n  {stats['filename']}:")
            for country, count in stats['country_dist'].most_common():
                pct = (count / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
                print(f"    {country:<20} | {count:>10,} ({pct:>5.2f}%)")

    # 6. ENTITY ID / DUPLICATE ANALYSIS
    print("\n[6] ENTITY ID / DUPLICATE ANALYSIS")
    print("-" * 100)
    for stats in all_stats:
        if 'unique_entity_id_count' in stats:
            print(f"\n  {stats['filename']}:")
            print(f"    Total rows: {stats['total_rows']:,}")
            print(f"    Unique entity_ids: {stats['unique_entity_id_count']:,}")
            print(f"    Duplicates found: {len(stats.get('duplicate_entity_ids', [])):,}")

            if stats['entity_id_prefixes']:
                print(f"    Entity ID prefixes:")
                for prefix, count in stats['entity_id_prefixes'].most_common():
                    print(f"      {prefix:<10} | {count:>10,}")

            if stats.get('duplicate_entity_ids'):
                print(f"    Sample duplicates: {stats['duplicate_entity_ids'][:5]}")

    # 7. GROUND TRUTH STRUCTURE
    print("\n[7] GROUND TRUTH STRUCTURE")
    print("-" * 100)
    gt_stats = [s for s in all_stats if 'ground_truth' in s['filename']]
    if gt_stats:
        gt = gt_stats[0]
        print(f"  Columns: {gt['column_order']}")
        print(f"  Total rows: {gt['total_rows']:,}")
        print(f"  Empty matches (singletons): {gt.get('empty_matches', 0):,}")
        print(f"  Unique Source 1 IDs: {gt.get('unique_source1_ids', 0):,}")
        print(f"  Unique matched S2 IDs: {gt.get('unique_matched_s2_ids', 0):,}")
        print(f"  Unique matched S3 IDs: {gt.get('unique_matched_s3_ids', 0):,}")

        if gt.get('match_distribution'):
            print(f"\n  Match distribution (number of matches per S1 entity):")
            for num_matches, count in sorted(gt['match_distribution'].items()):
                print(f"    {num_matches} matches: {count:,} entities")

        print(f"\n  Sample rows:")
        for i, row in enumerate(gt['sample_rows'][:3], 1):
            print(f"    [{i}] {row}")

    # 8. TRAIN VS TEST STRUCTURE
    print("\n[8] TRAIN VS TEST STRUCTURE")
    print("-" * 100)
    train_files = [s for s in all_stats if 'train_source' in s['filename']]
    test_files = [s for s in all_stats if 'test_source' in s['filename']]

    print(f"  Training files: {len(train_files)}")
    print(f"  Test files: {len(test_files)}")

    print(f"\n  Schema comparison:")
    for i in range(1, 4):
        train = [s for s in train_files if f'source{i}' in s['filename']]
        test = [s for s in test_files if f'source{i}' in s['filename']]

        if train and test:
            t = train[0]
            te = test[0]
            schema_match = t['column_order'] == te['column_order']
            print(f"    Source {i}: Schema match = {schema_match}")
            print(f"      Train: {t['column_order']}")
            print(f"      Test:  {te['column_order']}")

    # 9. SAMPLE DATA
    print("\n[9] SAMPLE DATA")
    print("-" * 100)
    for stats in all_stats:
        if stats.get('sample_rows'):
            print(f"\n  {stats['filename']} (first 2 rows):")
            for i, row in enumerate(stats['sample_rows'][:2], 1):
                print(f"    [{i}] {row}")

    # 10. DATA QUALITY RISKS
    print("\n[10] DATA QUALITY RISKS")
    print("-" * 100)
    issues = []

    for stats in all_stats:
        if stats.get('parsing_errors'):
            issues.append(f"  ⚠ {stats['filename']}: Parsing errors detected - {stats['parsing_errors']}")

        if stats.get('duplicate_entity_ids'):
            issues.append(f"  ⚠ {stats['filename']}: Contains {len(stats['duplicate_entity_ids'])} duplicate entity_ids")

        if 'missing_counts' in stats:
            for col, count in stats['missing_counts'].items():
                pct = (count / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
                if pct > 10:
                    issues.append(f"  ⚠ {stats['filename']}: High missing rate in '{col}' ({pct:.1f}%)")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No critical data quality issues detected")

    print("\n" + "="*100)

def main():
    # Define file paths
    base_dir = "D:\\Amazon_ML_Challenge_2026\\student_resource"

    files_to_inspect = [
        # Training files
        os.path.join(base_dir, "dataset", "train", "train_source1.tsv"),
        os.path.join(base_dir, "dataset", "train", "train_source2.tsv"),
        os.path.join(base_dir, "dataset", "train", "train_source3.tsv"),
        os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv"),
        # Test files
        os.path.join(base_dir, "dataset", "test", "test_source1.tsv"),
        os.path.join(base_dir, "dataset", "test", "test_source2.tsv"),
        os.path.join(base_dir, "dataset", "test", "test_source3.tsv"),
    ]

    all_stats = []

    # Inspect each file
    for filepath in files_to_inspect:
        if not os.path.exists(filepath):
            print(f"WARNING: File not found - {filepath}")
            continue

        if 'ground_truth' in filepath:
            stats = inspect_ground_truth(filepath)
        else:
            stats = inspect_tsv_file(filepath)

        all_stats.append(stats)

    # Generate report
    print_report(all_stats)

    print("\n✓ Audit complete")

if __name__ == "__main__":
    main()
