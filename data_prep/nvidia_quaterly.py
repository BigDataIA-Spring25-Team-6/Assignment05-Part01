import os
import re
import time
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from s3_utils import upload_file_to_s3, generate_s3_object_key

# Load environment variables from .env
load_dotenv()

# NVIDIA Quarterly Results Page URL
URL = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"

# Temporary download directory inside the current working directory
TEMP_DOWNLOAD_DIR = os.path.join(os.getcwd(), "temp_downloads")
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

# Initialize Selenium WebDriver
driver = webdriver.Chrome()
driver.get(URL)

# Define a wait instance
wait = WebDriverWait(driver, 10)

# Define the years to check
YEARS_TO_CHECK = ["2020", "2021", "2022", "2023", "2024", "2025"]

# Function to clean filenames and remove unnecessary text
def clean_filename(text, year):
    text = re.sub(r"\(opens in new window\)", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"Q{year}", "", text)
    text = re.sub(r"[<>:\"/\\|?*]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    return text

# Function to download a PDF file locally
def download_pdf(pdf_url, local_path):
    try:
        print(f"Downloading: {pdf_url}")
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved locally: {local_path}")
            return True
        else:
            print(f"Failed to download {pdf_url}, Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False

results = []

# Loop through each desired year
for year in YEARS_TO_CHECK:
    try:
        year_dropdown = wait.until(EC.presence_of_element_located((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")))
        year_select = Select(year_dropdown)
        year_select.select_by_value(year)
        print(f"Processing year: {year}")
    except Exception as e:
        print(f"Could not select year {year}: {e}")
        continue

    time.sleep(2)

    try:
        accordion_container = driver.find_element(By.ID, "_ctrl0_ctl75_divFinancialAccordionItemsContainer")
    except Exception as e:
        print(f"Accordion container not found for year {year}: {e}")
        continue

    quarter_headers = accordion_container.find_elements(By.XPATH, ".//button")
    for header in quarter_headers:
        try:
            if header.get_attribute("aria-expanded") != "true":
                driver.execute_script("arguments[0].scrollIntoView(true);", header)
                header.click()
                time.sleep(1)
        except Exception as e:
            print(f"Error clicking quarter header '{header.text.strip()}' for year {year}: {e}")

    time.sleep(2)

    pdf_links = accordion_container.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]")
    for link in pdf_links:
        try:
            raw_text = link.text.strip()
            pdf_url = link.get_attribute("href")

            link_text = clean_filename(raw_text, year)

            if re.search(r"(10-K|10-Q|Form 10-Q)", link_text, re.IGNORECASE):
                try:
                    quarter = link.find_element(By.XPATH, "preceding::button[1]").text.strip()
                except Exception:
                    quarter = "Unknown"

                quarter = re.sub(r"Quarter (\d+)", r"Q\1", quarter).replace("Fourth", "Q4").replace("Third", "Q3").replace("Second", "Q2").replace("First", "Q1")
                quarter = re.sub(rf"\s*Q{year}\s*", "", quarter).strip()
                quarter = f"{year}{quarter}"

                filename = f"{quarter}_{link_text}.pdf"
                filename = filename.replace(f"_{year}Q", "_")
                local_path = os.path.join(TEMP_DOWNLOAD_DIR, filename)

                # Generate structured S3 path
                s3_path = generate_s3_object_key(quarter, "pdfs", filename)

                if download_pdf(pdf_url, local_path):
                    upload_file_to_s3(local_path, quarter)
                    os.remove(local_path)  # Delete after upload

                results.append({
                    "year": year,
                    "quarter": quarter,
                    "name": link_text,
                    "url": pdf_url,
                    "s3_path": s3_path
                })
        except Exception as e:
            print(f"Error processing PDF for year {year}: {e}")

driver.quit()

# print("Found filings:")
# for item in results:
#     print(f"Year: {item['year']} | Quarter: {item['quarter']} | Name: {item['name']} | URL: {item['url']} | S3: {item['s3_path']}")

# Cleanup temp folder
if os.path.exists(TEMP_DOWNLOAD_DIR) and not os.listdir(TEMP_DOWNLOAD_DIR):
    os.rmdir(TEMP_DOWNLOAD_DIR)
    print(f"Deleted temp folder: {TEMP_DOWNLOAD_DIR}")