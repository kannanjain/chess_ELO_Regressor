import csv
from pathlib import Path
import random
from collections import defaultdict


def main():
    input_path = '15_fen_popular.csv'
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    print(f"Loading {input_path}...")
    
    rows_by_family = defaultdict(list)
    fieldnames = None
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            opening_family = row.get('OpeningFamily', 'Unknown')
            rows_by_family[opening_family].append(row)
    
    print(f"Found {len(rows_by_family)} opening families")
    
    train_rows = []
    test_rows = []
    
    for family, rows in rows_by_family.items():
        random.shuffle(rows)
        
        # Calculate split point for 80 20
        split_point = int(len(rows) * 0.8)
        
        train_rows.extend(rows[:split_point])
        test_rows.extend(rows[split_point:])
        
        print(f"{family}: {len(rows)} games -> Train: {split_point}, Test: {len(rows) - split_point}")
    
    random.shuffle(train_rows)
    random.shuffle(test_rows)
    
    print(f"\nWriting {len(train_rows)} rows to train.csv")
    with open(train_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    
    # Write test.csv
    print(f"Writing {len(test_rows)} rows to test.csv")
    with open(test_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    
    print(f"\nSplit complete!")
    total_games = len(train_rows) + len(test_rows)
    if total_games > 0:
        print(f"Train: {len(train_rows)} games ({len(train_rows)/total_games*100:.1f}%)")
        print(f"Test: {len(test_rows)} games ({len(test_rows)/total_games*100:.1f}%)")
    else:
        print("No games to split!")


if __name__ == "__main__":
    main()

