
import pandas as pd
import numpy as np
from pathlib import Path



piece_map = {
    'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,  # Black pieces
    'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12  # White pieces
}

# Piece values for material calculation
piece_values = {
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,  # White pieces
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0   # Black pieces
}


def fen_to_tensor(fen):
    board_tensor = np.zeros((12, 8, 8), dtype=int)
    board_state = fen.split()[0].split('/')
    
    for rank_idx, row in enumerate(board_state):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece_type = piece_map.get(char, 0)
                if piece_type != 0:
                    board_tensor[piece_type - 1, rank_idx, file_idx] = 1
                file_idx += 1
    
    return board_tensor


def calculate_material_points(fen_string):
    board_state = fen_string.split()[0].split('/')
    
    white_points = 0
    black_points = 0
    
    for row in board_state:
        for char in row:
            if char.isdigit():
                continue
            if char in piece_values:
                if char.isupper():  # White piece
                    white_points += piece_values[char]
                else:  # Black piece
                    black_points += piece_values[char]
    
    return white_points, black_points




def extract_features_from_csv(input_csv, output_csv, fen_column='fen_move_10'):
    
    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_path)
    
    if fen_column not in df.columns:
        print(f"Error: Column '{fen_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Extracting features from {fen_column}...")
    
    feature_rows = []
    rows_processed = 0
    rows_failed = 0
    
    for idx, row in df.iterrows():
        rows_processed += 1
        if rows_processed % 1000 == 0:
            print(f"Processed {rows_processed} rows... (failed: {rows_failed})")
        
        fen_string = row[fen_column]
        
        if pd.isna(fen_string) or not fen_string:
            rows_failed += 1
            continue
        
        try:
            tensor = fen_to_tensor(fen_string)
            # Flatten to 768 features (12 × 8 × 8)
            flattened = tensor.flatten()
            
           
            white_points, black_points = calculate_material_points(fen_string)
            
            
            feature_row = row.to_dict()
            for i, val in enumerate(flattened):
                feature_row[f'feature_{i}'] = int(val)
            
           
            feature_row['white_material_points'] = white_points
            feature_row['black_material_points'] = black_points
    
            
            feature_rows.append(feature_row)
        except Exception:
            rows_failed += 1
            continue
    
    print(f"\nExtraction complete!")
    print(f"Total rows processed: {rows_processed}")
    print(f"Rows with features: {len(feature_rows)}")
    print(f"Rows failed: {rows_failed}")
    
    if len(feature_rows) > 0:
        features_df = pd.DataFrame(feature_rows)
        features_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"Total features: {len(features_df.columns)} (768 tensor features + 2 material point features + 1 stockfish evaluation + original columns)")
        return features_df
    else:
        print("No features extracted!")
        return None


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("Extracting features from train.csv...")
    print("="*60)
    extract_features_from_csv(
        script_dir / 'train.csv',
        script_dir / 'train_features.csv',
        fen_column='fen_move_10'
    )
    
    print("\n" + "="*60)
    print("Extracting features from test.csv")
    print("="*60)
    extract_features_from_csv(
        script_dir / 'test.csv',
        script_dir / 'test_features.csv',
        fen_column='fen_move_10'
    )
