# clean_data.py

import pandas as pd
from utils.text_cleaner import TextCleaner

def main():
    data_path = "/home/faiz/Documents/github/Customer_Support/data/enfuce_support_tickets_synthetic.jsonl"
    data = pd.read_json(data_path, lines=True)

    print(data.head())
    print(f"\nDataset shape: {data.shape}")

    # Initialize cleaner and process data
    cleaner = TextCleaner()
    cleaned_data = cleaner.process_dataframe(data)

    print("Cleaning completed!")
    print(f"Language distribution:\n{cleaned_data['detected_language'].value_counts()}")

    # Save cleaned data
    cleaned_data.to_csv('/home/faiz/Documents/github/Customer_Support/data/customer_support_tickets.csv', index=False)
    print("Cleaned data saved to CSV!")

if __name__ == "__main__":
    main()