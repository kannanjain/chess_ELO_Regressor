import csv
from pathlib import Path
from collections import Counter


def main():
    popular_openings = {
        "Queen's Pawn",
        "King's Pawn Opening",
        "Sicilian Defence",
        "Polish Opening",
        "French Defence",
        "Scandinavian Defence",
        "Queen's Gambit",
        "Caro-Kann Defence",
        "Bishop's Opening",
        "Philidor's Defence",
        "English Opening",
        "Pirc Defence",
        "Italian Game",
        "Reti Opening",
        "Petrov's Defence",
        "Modern Defence (Robatsch)",
        "Vienna Game",
        "Nimzovich-Larsen Attack",
        "Ruy Lopez",
        "Two Knights Defence",
        "Centre Game"
    }
    
    input_path = Path(__file__).parent / 'opening_games.csv'
    output_path = Path(__file__).parent / 'popular_opening_games.csv'
    
    print(f"Loading {input_path}")
    
    filtered_rows = []
    opening_counts = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            opening_family = row.get('OpeningFamily', '')
            if opening_family in popular_openings:
                filtered_rows.append(row)
                opening_counts[opening_family] += 1
    
    print(f"Filtered to {len(filtered_rows)} games from popular opening families")
    print(f"\nOpening family distribution in filtered data:")
    for opening, count in opening_counts.most_common():
        print(f"  {opening}: {count}")
    
   
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"\nSaved to {output_path}")
    print(f"Total columns: {len(fieldnames)}")
    print(f"Total rows: {len(filtered_rows)}")
    
    return filtered_rows


if __name__ == "__main__":
    main()

