
import pandas as pd
import kagglehub
from pathlib import Path
import duckdb


def main():
    opening_families = {
        "Polish Opening": "A00",
        "Nimzovich-Larsen Attack": "A01",
        "Bird's Opening": "A02-A03",
        "Reti Opening": "A04-A09",
        "English Opening": "A10-A39",
        "Queen's Pawn": ["A40-A41", "A45-A46", "A50", "D00", "D02", "D04-D05", "E00", "E10"],
        "Modern Defence": "A42",
        "Old Benoni Defence": "A43-A44",
        "Queen's Indian Defence": ["A47", "E12-E19"],
        "King's Indian Defence": ["A48-A49", "E60-E99"],
        "Budapest Defence": "A51-A52",
        "Old Indian Defence": "A53-A55",
        "Benoni Defence": ["A56", "A60-A79"],
        "Benko Gambit": "A57-A59",
        "Dutch Defence": "A80-A99",
        "King's Pawn Opening": ["B00", "C20", "C40", "C44"],
        "Scandinavian Defence": "B01",
        "Alekhine's Defence": "B02-B05",
        "Modern Defence (Robatsch)": "B06",
        "Pirc Defence": "B07-B09",
        "Caro-Kann Defence": "B10-B19",
        "Sicilian Defence": "B20-B99",
        "French Defence": "C00-C19",
        "Centre Game": "C21-C22",
        "Bishop's Opening": "C23-C24",
        "Vienna Game": "C25-C29",
        "King's Gambit": "C30-C39",
        "Philidor's Defence": "C41",
        "Petrov's Defence": "C42-C43",
        "Scotch Game": "C45",
        "Three Knights Game": "C46",
        "Four Knights Game": "C47-C49",
        "Italian Game": "C50",
        "Evans Gambit": "C51-C52",
        "Giuoco Piano": "C53-C54",
        "Two Knights Defence": "C55-C59",
        "Ruy Lopez": "C60-C99",
        "Richter-Veresov Attack": "D01",
        "Torre Attack": "D03",
        "Queen's Gambit": ["D06", "D07-D09", "D10-D19", "D20-D29", "D30-D42", "D43-D49", "D50-D69"],
        "Neo-Grünfeld Defence": "D70-D79",
        "Grünfeld Defence": "D80-D99",
        "Catalan Opening": "E01-E09",
        "Bogo-Indian Defence": "E11",
        "Nimzo-Indian Defence": "E20-E59"
    }
    
    print("Loading dataset")
    path = kagglehub.dataset_download("adityajha1504/chesscom-user-games-60000-games")
    csv_path = Path(path) / "club_games_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} games")
    
    print("Extracting ECO codes from PGN")
    df['Eco'] = df['pgn'].apply(
        lambda x: x.split('\n')[-15].split('"')[1] if len(x.split('\n')) > 14 else None
    )
    
   
    con = duckdb.connect(database=':memory:', read_only=False)
    con.register('games', df)
    

    print("Mapping ECO codes to opening families...")
    case_statements = []
    for family, ranges in opening_families.items():
        conditions = []
        if isinstance(ranges, list):
            for r in ranges:
                if '-' in r:
                    start, end = r.split('-')
                    conditions.append(f"(Eco >= '{start}' AND Eco <= '{end}')")
                else:
                    conditions.append(f"Eco = '{r}'")
        else:
            if '-' in ranges:
                start, end = ranges.split('-')
                conditions.append(f"(Eco >= '{start}' AND Eco <= '{end}')")
            else:
                conditions.append(f"Eco = '{ranges}'")
        
        family_escaped = family.replace("'", "''")
        case_statements.append(f"WHEN {' OR '.join(conditions)} THEN '{family_escaped}'")
    
    query = f"""
    SELECT 
        *,
        CASE
            {' '.join(case_statements)}
            ELSE 'Other'
        END as OpeningFamily
    FROM games
    """
    
    result_df = con.execute(query).fetchdf()
    
    print(f"Mapped {len(result_df)} games to opening families")
    print(f"\nOpening family distribution:")
    print(result_df['OpeningFamily'].value_counts().head(10))
    

    output_path = Path(__file__).parent / 'opening_games.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total columns: {len(result_df.columns)}")
    print(f"Total rows: {len(result_df)}")
    
    return result_df


if __name__ == "__main__":
    main()

