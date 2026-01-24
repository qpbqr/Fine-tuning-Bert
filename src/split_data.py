import pandas as pd
import os 
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def split_data_by_ticker():
    """
    Split data by ticker into train, validation, and test sets.
    - For each ticker, take 60% of negative news for training.
    - Take the same number of positive news for training.
    - The remaining 40% of negative news: 20% for validation, 20% for test.
    - The remaining positive news is also allocated to validation and test sets in proportion.
    """
    print("Loading data...")
    data_path = config.BALANCED_DATA_PATH

    df = pd.read_csv(data_path)
    train_data = []
    val_data = []
    test_data = []

    np.random.seed(config.RANDOM_SEED)

    tickers = sorted(df['ticker'].unique())
    total_tickers = len(tickers)
    print(f"Total tickers: {total_tickers}")

    for idx, ticker in enumerate(tickers, 1):
        ticker_data = df[df['ticker'] == ticker].copy()
        neg_data = ticker_data[ticker_data['label'] == 0].copy()
        pos_data = ticker_data[ticker_data['label'] == 1].copy()
        
        neg_count = len(neg_data)
        pos_count = len(pos_data)
        
        neg_train_size = int(neg_count * 0.6)
        neg_remaining = neg_count - neg_train_size
        # shuffle the data
        neg_data = neg_data.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        # split negative data into train val and test
        neg_train = neg_data.iloc[:neg_train_size].copy()
        neg_val = neg_data.iloc[neg_train_size:neg_train_size + neg_remaining//2].copy()
        neg_test = neg_data.iloc[neg_train_size + neg_remaining//2:].copy()
        # split positive data into train val and test
        pos_data = pos_data.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        pos_train = pos_data.iloc[:neg_train_size].copy()
        pos_remaining = pos_data.iloc[neg_train_size:].copy()
        pos_remaining_count = len(pos_remaining)
        if pos_remaining_count > 0:
            pos_val_size = neg_remaining // 2
            pos_val = pos_remaining.iloc[:pos_val_size].copy()
            pos_test = pos_remaining.iloc[pos_val_size:2*pos_val_size].copy()
        else:
            pos_val = pd.DataFrame()
            pos_test = pd.DataFrame()
        train_data.append(pd.concat([neg_train, pos_train], ignore_index=True))
        val_data.append(pd.concat([neg_val, pos_val], ignore_index=True))
        test_data.append(pd.concat([neg_test, pos_test], ignore_index=True))
        # print progress
        if idx % 50 == 0 or ticker in ['A', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']:
            print(f"[{idx}/{total_tickers}] {ticker}: "
                  f"Train set (negative/positive): {len(neg_train)}/{len(pos_train)}, "
                  f"Validation set (negative/positive): {len(neg_val)}/{len(pos_val)}, "
                  f"Test set (negative/positive): {len(neg_test)}/{len(pos_test)}")
        
# merge all ticker data
    print("\nMerging data...")
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # shuffle the final dataset
    train_df = train_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # ensure the output directory exists
    output_dir = os.path.dirname(config.TRAIN_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # save data
    train_path = config.TRAIN_DATA_PATH
    val_path = config.VAL_DATA_PATH
    test_path = config.TEST_DATA_PATH
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nData split completed!")
    print(f"Train set: {len(train_df)} (positive: {len(train_df[train_df['label']==1])}, "
          f"negative: {len(train_df[train_df['label']==0])})")
    print(f"Validation set: {len(val_df)} (positive: {len(val_df[val_df['label']==1])}, "
          f"negative: {len(val_df[val_df['label']==0])})")
    print(f"Test set: {len(test_df)} (positive: {len(test_df[test_df['label']==1])}, "
          f"negative: {len(test_df[test_df['label']==0])})")
    
    # save statistics
    stats_summary = {
        'dataset': ['train', 'val', 'test'],
        'total': [len(train_df), len(val_df), len(test_df)],
        'positive': [
            len(train_df[train_df['label']==1]),
            len(val_df[val_df['label']==1]),
            len(test_df[test_df['label']==1])
        ],
        'negative': [
            len(train_df[train_df['label']==0]),
            len(val_df[val_df['label']==0]),
            len(test_df[test_df['label']==0])
        ]
    }
    stats_summary_df = pd.DataFrame(stats_summary)
    stats_summary_path = os.path.join(output_dir, "split_stats.csv")
    stats_summary_df.to_csv(stats_summary_path, index=False)
    print(f"\nStatistics saved to: {stats_summary_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = split_data_by_ticker()