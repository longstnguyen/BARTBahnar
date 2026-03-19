import pandas as pd
import os
import glob


class DataProcessor:
    def __init__(self, input_dir, output_dir="../data"):
        """
        Initialize with a directory of CSV files and an output directory.
        Reads all .csv files found in input_dir.
        """
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
        if not self.input_paths:
            raise FileNotFoundError(f"No CSV files found in '{input_dir}'")
        self.output_dir = output_dir
        self.dataframes = []
        self.rows_per_file = []
        self.rows_removed = 0
        self.removed_rows_info = []
        self.duplicates_removed = 0
        self.merged_df = None

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_clean_data(self):
        """
        Read local CSV files, remove invalid rows, and deduplicate.
        """
        for path in self.input_paths:
            df = pd.read_csv(path)

            if "tiếng bana" in df.columns and "tiếng việt" in df.columns:
                initial_rows = df.shape[0]
                self.rows_per_file.append(initial_rows)

                # Find rows where exactly one column is missing
                invalid_rows = df[(df["tiếng bana"].isna() & ~df["tiếng việt"].isna()) |
                                  (~df["tiếng bana"].isna() & df["tiếng việt"].isna())]

                self.rows_removed += invalid_rows.shape[0]
                self.removed_rows_info.append((path, invalid_rows.index.tolist()))

                # Drop invalid rows
                df = df.drop(invalid_rows.index)
                self.dataframes.append(df[["tiếng bana", "tiếng việt"]])

        # Merge all DataFrames together
        self.merged_df = pd.concat(self.dataframes, ignore_index=True)

        # Remove duplicate records
        initial_merged_rows = self.merged_df.shape[0]
        self.merged_df = self.merged_df.drop_duplicates(subset=["tiếng bana"]).drop_duplicates(subset=["tiếng việt"])
        self.duplicates_removed = initial_merged_rows - self.merged_df.shape[0]

    def save_clean_data(self, output_filename="final.csv"):
        """
        Save the cleaned data to a CSV file in the output directory.
        """
        output_path = os.path.join(self.output_dir, output_filename)
        if self.merged_df is not None:
            self.merged_df.to_csv(output_path, index=False)
            # print(f"✅ Clean data saved to {output_path}")
        else:
            print("No data to save.")

    def extract_sentences(self, column_name="tiếng bana", output_filename="bana_data.txt"):
        """
        Extract sentences from a specific column and save them to a text file.
        """
        if self.merged_df is None or column_name not in self.merged_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist or data has not been loaded.")

        # Normalize unicode and strip whitespace
        self.merged_df[column_name] = self.merged_df[column_name].str.normalize('NFKC').str.strip()

        # Filter unique sentences
        sentences = self.merged_df[column_name].dropna().str.strip().unique()

        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in sentences:
                if sentence:
                    f.write(sentence + "\n")

        # print(f"✅ Saved {len(sentences)} sentences to '{output_path}'")

    def print_summary(self):
        """
        Print a summary of the data processing results.
        """
        # print("📊 Data processing summary:")
        # print("Total files:", len(self.input_paths))
        # print("Rows per file (before cleaning):", self.rows_per_file)
        # print("Rows removed (missing one column):", self.rows_removed)
        # print("Duplicate rows removed:", self.duplicates_removed)
        # print("Rows remaining after processing:", self.merged_df.shape[0] if self.merged_df is not None else 0)
        # print("Addresses of removed rows:")
        # for file, rows in self.removed_rows_info:
        #     print(f"File: {file}, Removed rows: {rows}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Clean and export raw Bahnar-Vietnamese CSV files."
    )
    parser.add_argument(
        "--input_dir", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "raw"),
        help="Directory containing raw CSV files (default: data/raw/)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data"),
        help="Directory to save output files (default: data/)"
    )
    args = parser.parse_args()

    processor = DataProcessor(input_dir=args.input_dir, output_dir=args.output_dir)
    processor.load_and_clean_data()
    processor.save_clean_data("final.csv")
    processor.extract_sentences("tiếng bana", "bahnaric.txt")
    processor.print_summary()

