import asyncio
import json

async def download_pdfs(db_path: str):
    with open(db_path, "r") as f:
        data_json = json.load(f)

    for data in data_json:
        print(data)
        break


if __name__ == "__main__":
    DB_PATH = "data/Release/Database/database.json"
    asyncio.run(download_pdfs(db_path=DB_PATH))
