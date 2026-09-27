#!/usr/bin/env python3
"""Quick targeted dataset audit — low memory, streaming reads."""

import csv
import os
import sys
from collections import Counter, defaultdict

BASE = r"D:\Amazon_ML_Challenge_2026\student_resource\dataset"
SEP = "=" * 100
DASH = "-" * 100

# =========================================================================
# PART 1: Sample rows from each source file
# =========================================================================
def part1_samples():
    print("\n" + SEP)
    print("PART 1: SAMPLE ROWS")
    print(SEP)
    for d in ["train", "test"]:
        dp = os.path.join(BASE, d)
        for f in sorted(os.listdir(dp)):
            fp = os.path.join(dp, f)
            print("\n" + DASH)
            print("%s/%s:" % (d, f))
            print(DASH)
            with open(fp, "r", encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter="\t")
                header = next(reader)
                print("  Header: %s" % header)
                for i, row in enumerate(reader):
                    if i >= 5:
                        break
                    print("  Row %d: %s" % (i + 1, row))

# =========================================================================
# PART 2: Missing / empty values per column (streaming)
# =========================================================================
def part2_missing():
    print("\n" + SEP)
    print("PART 2: MISSING / EMPTY VALUES")
    print(SEP)
    for d in ["train", "test"]:
        dp = os.path.join(BASE, d)
        for f in sorted(os.listdir(dp)):
            fp = os.path.join(dp, f)
            print("\n" + DASH)
            print("%s/%s:" % (d, f))
            total = 0
            missing = defaultdict(int)  # column -> null/missing count
            empty = defaultdict(int)    # column -> empty-string count
            with open(fp, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                cols = reader.fieldnames
                for row in reader:
                    total += 1
                    for col in cols:
                        val = row.get(col)
                        if val is None:
                            missing[col] += 1
                        elif val.strip() == "":
                            empty[col] += 1
            print("  Total rows: %s" % f"{total:,}")
            for col in cols:
                m = missing.get(col, 0)
                e = empty.get(col, 0)
                m_pct = (m / total * 100) if total else 0
                e_pct = (e / total * 100) if total else 0
                print("    %-25s  Missing: %8s (%5.2f%%)  Empty: %8s (%5.2f%%)" % (
                    col, f"{m:,}", m_pct, f"{e:,}", e_pct))

# =========================================================================
# PART 3: Country distribution
# =========================================================================
def part3_country():
    print("\n" + SEP)
    print("PART 3: COUNTRY DISTRIBUTION")
    print(SEP)
    for d in ["train", "test"]:
        dp = os.path.join(BASE, d)
        for f in sorted(os.listdir(dp)):
            if "ground_truth" in f:
                continue
            fp = os.path.join(dp, f)
            print("\n%s/%s:" % (d, f))
            country_counts = Counter()
            total = 0
            with open(fp, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    total += 1
                    c = row.get("country", "").strip()
                    if c:
                        country_counts[c] += 1
                    else:
                        country_counts["<EMPTY/MISSING>"] += 1
            for country, cnt in country_counts.most_common():
                pct = cnt / total * 100
                print("    %-25s  %10s (%5.2f%%)" % (country, f"{cnt:,}", pct))

# =========================================================================
# PART 4: Entity ID analysis (prefixes, uniqueness, overlap)
# =========================================================================
def part4_entity_ids():
    print("\n" + SEP)
    print("PART 4: ENTITY ID ANALYSIS")
    print(SEP)

    # Collect entity IDs per file
    for d in ["train", "test"]:
        dp = os.path.join(BASE, d)
        for f in sorted(os.listdir(dp)):
            if "ground_truth" in f:
                continue
            fp = os.path.join(dp, f)
            print("\n%s/%s:" % (d, f))
            prefix_counts = Counter()
            total = 0
            dup_count = 0
            seen = set()
            sample_ids = []
            with open(fp, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    total += 1
                    eid = row.get("entity_id", "").strip()
                    if eid:
                        if eid in seen:
                            dup_count += 1
                        else:
                            seen.add(eid)
                        if "-" in eid:
                            prefix_counts[eid.split("-")[0]] += 1
                        if len(sample_ids) < 5:
                            sample_ids.append(eid)
            print("    Total rows: %s" % f"{total:,}")
            print("    Unique entity_ids: %s" % f"{len(seen):,}")
            print("    Duplicate entity_ids: %s" % f"{dup_count:,}")
            print("    Sample IDs: %s" % sample_ids)
            print("    Prefix distribution:")
            for prefix, cnt in prefix_counts.most_common():
                print("      %-10s  %10s" % (prefix, f"{cnt:,}"))

# =========================================================================
# PART 5: Ground truth analysis
# =========================================================================
def part5_ground_truth():
    print("\n" + SEP)
    print("PART 5: GROUND TRUTH ANALYSIS")
    print(SEP)

    fp = os.path.join(BASE, "train", "train_ground_truth.tsv")
    total = 0
    empty_matches = 0
    match_count_dist = Counter()
    s2_matches = 0
    s3_matches = 0
    total_match_ids = 0
    sample_rows = []

    with open(fp, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            total += 1
            matched = row.get("matched_entity_ids", "").strip()

            if total <= 10:
                sample_rows.append(row)

            if not matched:
                empty_matches += 1
                match_count_dist[0] += 1
            else:
                ids = [m.strip() for m in matched.split(",") if m.strip()]
                match_count_dist[len(ids)] += 1
                total_match_ids += len(ids)
                for mid in ids:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1

    print("  Total rows: %s" % f"{total:,}")
    print("  Singletons (empty matches): %s (%5.2f%%)" % (
        f"{empty_matches:,}", empty_matches / total * 100 if total else 0))
    print("  Total match IDs referenced: %s" % f"{total_match_ids:,}")
    print("  S2 match references: %s" % f"{s2_matches:,}")
    print("  S3 match references: %s" % f"{s3_matches:,}")

    print("\n  Match count distribution (matches per S1 entity):")
    for n in sorted(match_count_dist.keys()):
        cnt = match_count_dist[n]
        pct = cnt / total * 100 if total else 0
        print("    %3d matches: %10s entities (%5.2f%%)" % (n, f"{cnt:,}", pct))

    print("\n  Sample ground truth rows:")
    for i, row in enumerate(sample_rows[:5]):
        s1 = row.get("source1_entity_id", "")
        matched = row.get("matched_entity_ids", "")
        print("    [%d] S1: %-15s  Matches: %s" % (i + 1, s1, matched if matched else "<empty>"))

# =========================================================================
# PART 6: Train ground-truth S1 IDs vs source1 IDs correspondence
# =========================================================================
def part6_id_correspondence():
    print("\n" + SEP)
    print("PART 6: ID CORRESPONDENCE CHECK")
    print(SEP)

    # Read source1 entity_ids
    source1_ids = set()
    fp = os.path.join(BASE, "train", "train_source1.tsv")
    with open(fp, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            source1_ids.add(row.get("entity_id", "").strip())

    # Read ground truth S1 ids
    gt_ids = set()
    fp = os.path.join(BASE, "train", "train_ground_truth.tsv")
    with open(fp, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gt_ids.add(row.get("source1_entity_id", "").strip())

    print("  Source1 entity_ids count: %s" % f"{len(source1_ids):,}")
    print("  Ground truth S1 entity_ids count: %s" % f"{len(gt_ids):,}")
    print("  IDs in GT but NOT in Source1: %s" % f"{len(gt_ids - source1_ids):,}")
    print("  IDs in Source1 but NOT in GT: %s" % f"{len(source1_ids - gt_ids):,}")
    if gt_ids == source1_ids:
        print("  >>> PERFECT 1:1 CORRESPONDENCE")
    else:
        print("  >>> MISMATCH FOUND")
        diff1 = gt_ids - source1_ids
        diff2 = source1_ids - gt_ids
        if diff1:
            print("    Sample GT-only: %s" % list(diff1)[:5])
        if diff2:
            print("    Sample Source1-only: %s" % list(diff2)[:5])

# =========================================================================
# PART 7: Check tab-consistency (column count per line)
# =========================================================================
def part7_tab_consistency():
    print("\n" + SEP)
    print("PART 7: STRUCTURAL CONSISTENCY (tab count per line)")
    print(SEP)
    for d in ["train", "test"]:
        dp = os.path.join(BASE, d)
        for f in sorted(os.listdir(dp)):
            fp = os.path.join(dp, f)
            with open(fp, "r", encoding="utf-8") as fh:
                header = fh.readline()
                expected_tabs = header.count("\t")
                total = 0
                bad_lines = []
                for line_num, line in enumerate(fh, start=2):
                    total += 1
                    actual_tabs = line.count("\t")
                    if actual_tabs != expected_tabs:
                        if len(bad_lines) < 5:
                            bad_lines.append((line_num, actual_tabs, line[:200]))
            if bad_lines:
                print("  %s/%s: %d rows with wrong tab count (expected %d tabs):" % (
                    d, f, len(bad_lines), expected_tabs))
                for ln, t, preview in bad_lines:
                    print("    Line %d: %d tabs -- '%s...'" % (ln, t, preview))
            else:
                print("  %s/%s: CONSISTENT (%d tabs per line, %s data rows)" % (
                    d, f, expected_tabs, f"{total:,}"))


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    part1_samples()
    print("\n\n*** Part 1 complete. Starting Part 2...", flush=True)
    part2_missing()
    print("\n\n*** Part 2 complete. Starting Part 3...", flush=True)
    part3_country()
    print("\n\n*** Part 3 complete. Starting Part 4...", flush=True)
    part4_entity_ids()
    print("\n\n*** Part 4 complete. Starting Part 5...", flush=True)
    part5_ground_truth()
    print("\n\n*** Part 5 complete. Starting Part 6...", flush=True)
    part6_id_correspondence()
    print("\n\n*** Part 6 complete. Starting Part 7...", flush=True)
    part7_tab_consistency()
    print("\n\nALL PARTS COMPLETE.")
