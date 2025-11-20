
import csv
from pathlib import Path
import chess
import chess.pgn


def extract_moves_from_pgn(pgn_text):

    try:
        pgn_text = pgn_text.strip()
        if not pgn_text:
            return None
        
        pgn_io = chess.pgn.read_game(chess.pgn.StringIO(pgn_text))
        if pgn_io is None:
            return None
        
        moves = []
        node = pgn_io
        
        while node.variations:
            node = node.variation(0)
            if node.move is not None:
                moves.append(node.move)
        
        return moves if moves else None
    except (ValueError, AttributeError) as e:
        try:
            lines = pgn_text.split('\n')
            move_text = ''
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('[') and not line.startswith('{'):
                    move_text = line
                    break
            
            if move_text:
                board = chess.Board()
                moves = []
                # Split by move numbers and extract moves
                parts = move_text.split()
                for part in parts:
                    
                    if part.endswith('.') or part in ['1-0', '0-1', '1/2-1/2', '*']:
                        continue
                    try:
                        move = board.parse_san(part)
                        board.push(move)
                        moves.append(move)
                        if len(moves) >= 15:
                            break
                    except:
                        continue
                
                return moves if len(moves) >= 15 else None
        except:
            pass
        return None
    except Exception as e:
        return None


def get_fen_after_moves(moves, num_moves=15):
    if len(moves) < num_moves:
        return None
    
    board = chess.Board()
    fen_positions = []
    
    for i in range(num_moves):
        try:
            board.push(moves[i])
            fen_positions.append(board.fen())
        except Exception:
            return None
    
    return fen_positions


def main():
    input_path = 'popular_opening_games.csv'
    output_path =  '15_fen_popular.csv'
    
    print(f"Loading {input_path}")
    
    rows_processed = 0
    rows_written = 0
    rows_dropped = 0
    drop_reasons = {'no_pgn': 0, 'parse_failed': 0, 'too_short': 0, 'fen_failed': 0}
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Prepare output
        output_fieldnames = list(fieldnames) + [f'fen_move_{i+1}' for i in range(15)]
        
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for row in reader:
                rows_processed += 1
                if rows_processed % 1000 == 0:
                    print(f"Processed {rows_processed} games (written: {rows_written}, dropped: {rows_dropped})")
                    if rows_processed == 1000:
                        print(f"  Drop reasons so far: {drop_reasons}")
                
                pgn_text = row.get('pgn', '')
                if not pgn_text:
                    rows_dropped += 1
                    drop_reasons['no_pgn'] += 1
                    continue
                
                moves = extract_moves_from_pgn(pgn_text)
                if moves is None:
                    rows_dropped += 1
                    drop_reasons['parse_failed'] += 1
                    if drop_reasons['parse_failed'] <= 3:
                        print(f"  Parse failed for game {rows_processed} (OpeningFamily: {row.get('OpeningFamily', 'N/A')})")
                    continue
                
                if len(moves) < 15:
                    rows_dropped += 1
                    drop_reasons['too_short'] += 1
                    continue
        
                fen_positions = get_fen_after_moves(moves, 15)
                if fen_positions is None:
                    rows_dropped += 1
                    drop_reasons['fen_failed'] += 1
                    continue

                for i, fen in enumerate(fen_positions):
                    row[f'fen_move_{i+1}'] = fen
                
                writer.writerow(row)
                rows_written += 1
                
                if rows_written == 1:
                    print(f"First successful game: {len(moves)} moves extracted")
    
    print(f"\nProcessing complete!")
    print(f"Total games processed: {rows_processed}")
    print(f"Games with 15+ moves: {rows_written}")
    print(f"Games dropped: {rows_dropped}")
    print(f"Drop reasons: {drop_reasons}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

