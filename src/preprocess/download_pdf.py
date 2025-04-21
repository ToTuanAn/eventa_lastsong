import asyncio
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import json
import glob
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import argparse

# service = Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service)

save_directory = "/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf"  # Replace with the desired directory

def url_to_pdf(url, output_pdf):
    # Set up Chrome options to run in headless mode
    options = webdriver.ChromeOptions()
    settings = {
        "recentDestinations": [{
            "id": "Save as PDF",
            "origin": "local",
            "account": "",
        }],
        "selectedDestinationId": "Save as PDF",
        "version": 2,
        "marginsType": 1,
        "mediaSize": {
            "name": "CUSTOM",
            "width_microns": 279400,  # 11 inches
            "height_microns": 431800  # 17 inches
        },
        "isHeaderFooterEnabled": False,
        "isLandscapeEnabled": False
    }
    prefs = {'printing.print_preview_sticky_settings.appState': json.dumps(settings),
             'savefile.default_directory': save_directory,
             'savefile.filename': output_pdf}
    options.add_experimental_option('prefs', prefs)
    options.add_argument('--kiosk-printing')
    options.add_argument('--disable-automation')
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.page_load_strategy = 'none'  # Skip full page load
    options.add_argument("--disable-blink-features=AutomationControlled")

    # Redirect ad.doubleclick.net to 127.0.0.1
    options.add_argument("--host-rules=MAP ad.doubleclick.net 127.0.0.1")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Open the URL
    driver.maximize_window()
    driver.get(url)
    # driver.execute_script("window.stop();")
    time.sleep(3)
    #
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # # Wait for the page to load (adjust as necessary for your case)
    driver.execute_script("""
        var ads = document.querySelectorAll('[class*="ad-slot"], [class*="gallery-inline"], [class*="container_list-headlines-with-read-times"], [class*="container_list-headlines-ranked"], [class*="layout__bottom"], [class*="container_list-headlines-with-images"], [class*="layout-with-rail__info layout-with-rail__topFullBleed"], [class*="header cnn-app-display-none"], [class*="headline__sub-text"], [class*="vossi-social-share"], [class*="related-content--article"], [class*="related-content--gallery"], [class*="market-feature-ribbon"], [class*="video-inline_carousel"], [class*="_dianomi_wrapper"]');
        ads.forEach(ad => ad.remove());
    """)

    driver.execute_script("window.stop();")

    time.sleep(3)

    # Trigger the print dialog programmatically (but it won't block because we are saving as PDF automatically)
    driver.execute_script('window.print();')

    # Wait for the file to be saved (adjust sleep time if needed)
    time.sleep(5)

    # Close the browser
    driver.quit()

def get_most_recent_file(directory):
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        return None  # or raise an exception
    recent_file = max(files, key=os.path.getctime)
    return recent_file


async def download_pdfs(db_path: str, pdf_output_root: str):
    with open(db_path, "r") as f:
        data_json = json.load(f)

    for article_key in tqdm(data_json):
        if os.path.isfile(os.path.join(pdf_output_root,f"{article_key}.pdf")):
            continue

        url_to_pdf(url=data_json[article_key]["url"],output_pdf=f"{article_key}.pdf")

        if not os.path.isfile(os.path.join(pdf_output_root,f"{article_key}.pdf")):
            recent_file = get_most_recent_file(pdf_output_root)
            os.rename(recent_file, f"{pdf_output_root}/{article_key}.pdf")

    return

def split_db(db_path, output_path):
    with open(db_path, "r") as f:
        data_json = json.load(f)

    split_json = {}
    idx = 1

    for article_key in tqdm(data_json):
        while len(split_json) > len(data_json) // 20:
            with open(f"{output_path}/splitted_database_{str(idx)}.json", "w") as f:
                json.dump(split_json, f, indent=4)
                split_json = {}
                idx += 1
        split_json[article_key] = data_json[article_key]

    with open(f"{output_path}/splitted_database_{str(idx)}.json", "w") as f:
        json.dump(split_json, f, indent=4)
        split_json = {}
        idx += 1
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple argparse example")

    # Add arguments
    parser.add_argument('--db_path', type=str, default="/home/totuanan/Workplace/eventa_lastsong/data/Release/Database_Splitted/splitted_database_1.json", help='Input file path')
    parser.add_argument('--output_path', type=str, default="/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf", help='Output file path')

    args = parser.parse_args()

    asyncio.run(download_pdfs(db_path=args.db_path, pdf_output_root=args.output_path))
    #
    # OUTPUT_PATH =  "/home/totuanan/Workplace/eventa_lastsong/data/Release/Database_Splitted"
    # split_db(DB_PATH, OUTPUT_PATH)
