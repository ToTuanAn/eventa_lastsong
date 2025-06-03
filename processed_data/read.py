import polars as pl

def read_parquet(file_path: str) -> pl.DataFrame:
    """
    Reads a Parquet file and returns a Polars DataFrame.
    
    :param file_path: Path to the Parquet file.
    :return: Polars DataFrame containing the data from the Parquet file.
    """
    return pl.read_parquet(file_path)

if __name__ == "__main__":
    # Example usage
    file_path = ".\\database_v4_chunk_emb.parquet"
    df = read_parquet(file_path)
    print(df.filter(pl.col("chunk_emb") == "").count())