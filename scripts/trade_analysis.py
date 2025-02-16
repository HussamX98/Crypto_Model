import os
import json
import pandas as pd
from datetime import datetime

def analyze_trades(trades_dir: str):
    """Analyze and combine all trade data from a session"""
    all_trades = []
    
    # Get all trade IDs from filenames
    files = os.listdir(trades_dir)
    trade_ids = set()
    for file in files:
        if file.startswith('TRADE_'):
            trade_id = '_'.join(file.split('_')[:-1])
            trade_ids.add(trade_id)
    
    # Process each trade
    for trade_id in sorted(trade_ids):  # Sort to process in order
        trade_data = {}
        
        # Load entry data
        entry_file = os.path.join(trades_dir, f"{trade_id}_entry.json")
        if os.path.exists(entry_file):
            with open(entry_file, 'r') as f:
                entry_data = json.load(f)
                trade_data.update({
                    'trade_id': trade_id,
                    'token_symbol': entry_data.get('token_symbol'),
                    'entry_time': entry_data.get('entry_time'),
                    'entry_price': entry_data.get('entry_price'),
                    'market_price_at_entry': entry_data.get('market_price'),
                    'position_size': entry_data.get('position_size'),
                    'confidence_score': entry_data.get('confidence_score'),
                    'entry_features': entry_data.get('features')
                })
        
        # Load exit data
        exit_file = os.path.join(trades_dir, f"{trade_id}_exit.json")
        if os.path.exists(exit_file):
            with open(exit_file, 'r') as f:
                exit_data = json.load(f)
                trade_data.update({
                    'exit_time': exit_data.get('exit_time'),
                    'exit_price': exit_data.get('exit_price'),
                    'return_pct': exit_data.get('return_pct', 0),  # Default to 0 if None
                    'time_in_trade_minutes': exit_data.get('time_in_trade_minutes'),
                    'exit_type': exit_data.get('exit_type'),
                    'peak_return': exit_data.get('peak_return', 0)  # Default to 0 if None
                })
        else:
            trade_data.update({
                'exit_time': 'Still Open',
                'exit_price': None,
                'return_pct': 0,  # Default to 0 for open trades
                'time_in_trade_minutes': None,
                'exit_type': 'Open',
                'peak_return': 0
            })
        
        # Load price action data
        price_action_file = os.path.join(trades_dir, f"{trade_id}_price_action.json")
        if os.path.exists(price_action_file):
            with open(price_action_file, 'r') as f:
                price_action_data = json.load(f)
                if isinstance(price_action_data, list):
                    # If price_action_data is a list, take max/min values
                    prices = [float(p.get('price', 0)) for p in price_action_data if p.get('price') is not None]
                    returns = [float(p.get('return_pct', 0)) for p in price_action_data if p.get('return_pct') is not None]
                    trade_data.update({
                        'highest_price': max(prices) if prices else 0,
                        'lowest_price': min(prices) if prices else 0,
                        'max_return': max(returns) if returns else 0,
                        'min_return': min(returns) if returns else 0,
                        'price_points_recorded': len(price_action_data)
                    })
                else:
                    # If price_action_data is a dict with summary data
                    trade_data.update({
                        'highest_price': price_action_data.get('highest_price_seen', 0),
                        'monitoring_duration': price_action_data.get('monitoring_duration_minutes', 0)
                    })
        
        all_trades.append(trade_data)
    
    # Calculate summary statistics for completed trades only
    completed_trades = [t for t in all_trades if t['exit_type'] != 'Open']
    
    summary_stats = {
        'total_trades': len(all_trades),
        'completed_trades': len(completed_trades),
        'open_trades': len(all_trades) - len(completed_trades),
        'profitable_trades': sum(1 for t in completed_trades if t['return_pct'] > 0),
        'stop_losses': sum(1 for t in completed_trades if 'stop_loss' in str(t.get('exit_type', ''))),
    }
    
    # Add average metrics only if there are completed trades
    if completed_trades:
        completed_returns = [t['return_pct'] for t in completed_trades]
        completed_durations = [t['time_in_trade_minutes'] for t in completed_trades if t['time_in_trade_minutes'] is not None]
        
        summary_stats.update({
            'average_return': sum(completed_returns) / len(completed_returns),
            'average_trade_duration': sum(completed_durations) / len(completed_durations) if completed_durations else 0,
            'best_trade': max(completed_returns),
            'worst_trade': min(completed_returns)
        })
    else:
        summary_stats.update({
            'average_return': 0,
            'average_trade_duration': 0,
            'best_trade': 0,
            'worst_trade': 0
        })
    
    # Calculate win rate
    if completed_trades:
        summary_stats['win_rate'] = (summary_stats['profitable_trades'] / len(completed_trades)) * 100
    else:
        summary_stats['win_rate'] = 0
    
    # Combine everything into final results
    final_results = {
        'summary_stats': summary_stats,
        'trades': all_trades
    }
    
    # Save to file
    output_file = os.path.join(os.path.dirname(trades_dir), 'trade_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\nTrade Analysis Summary:")
    print(f"Total Trades: {summary_stats['total_trades']}")
    print(f"Completed Trades: {summary_stats['completed_trades']}")
    print(f"Open Trades: {summary_stats['open_trades']}")
    print(f"Profitable Trades: {summary_stats['profitable_trades']}")
    print(f"Stop Losses: {summary_stats['stop_losses']}")
    print(f"Win Rate: {summary_stats['win_rate']:.2f}%")
    print(f"Average Return: {summary_stats['average_return']:.2f}%")
    print(f"Average Trade Duration: {summary_stats['average_trade_duration']:.2f} minutes")
    print(f"Best Trade Return: {summary_stats['best_trade']:.2f}%")
    print(f"Worst Trade Return: {summary_stats['worst_trade']:.2f}%")
    
    # Print individual trade summaries
    print("\nIndividual Trade Summary:")
    for trade in all_trades:
        print(f"\nTrade ID: {trade['trade_id']}")
        print(f"Token: {trade['token_symbol']}")
        print(f"Status: {'Completed' if trade['exit_type'] != 'Open' else 'Open'}")
        if trade['exit_type'] != 'Open':
            print(f"Return: {trade['return_pct']:.2f}%")
            print(f"Exit Type: {trade['exit_type']}")
            print(f"Duration: {trade['time_in_trade_minutes']:.1f} minutes")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    trades_dir = r"C:\Users\alsal\Projects\Hussam\crypto_model\output\20250211_105802\trades"
    analyze_trades(trades_dir)