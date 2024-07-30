import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
from PIL import Image
from io import BytesIO

def scrape_images(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    urls = [img['src'] for img in img_tags if 'drone' in img.get('alt', '').lower()]
    return urls

def scrape_images_selenium(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)
    img_elements = driver.find_elements(By.TAG_NAME, 'img')
    urls = [img.get_attribute('src') for img in img_elements if 'drone' in img.get_attribute('alt').lower()]
    driver.quit()
    return urls

def download_images(image_urls, save_dir="C:\\Users\\lburns\\Desktop\\TESTCAPSTONE\\testscraper"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, url in enumerate(image_urls):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(save_dir, f'drone_{i}.jpg'))


if __name__ == "__main__":
    
    url = 'https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle' 
    drone_images = scrape_images_selenium(url)
    print(drone_images)
    download_images(drone_images)