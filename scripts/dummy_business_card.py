import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from random import randint
from urllib.request import urlopen

import numpy as np
import requests
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from src.domain.business_card import BusinessCard

DATASET_PATH = Path("dataset/validation")
LABEL_PATH = DATASET_PATH / "label.json"
IMAGE_DIR = DATASET_PATH / "image"
IMAGE_SIZE = (700, 500)
PICSUM_URL = f"https://picsum.photos/{IMAGE_SIZE[0]}/{IMAGE_SIZE[1]}"
FONT_PATH = "https://github.com/googlefonts/morisawa-biz-ud-mincho/raw/main/fonts/ttf/BIZUDPMincho-Regular.ttf"
DATASET_LENGTH = 100

TIMEOUT = 1000


def color_invert(r: int, g: int, b: int) -> str:
    mono = (0.114 * r) + (0.587 * g) + (0.299 * b)
    if mono >= 127:
        return "#000000"

    return "#FFFFFF"


def fetch_image_from_url(url: str) -> Image.Image | None:
    try:
        image = Image.open(BytesIO(requests.get(url, timeout=TIMEOUT).content))
    except (TypeError, ValueError, ConnectionError, OSError, BufferError):
        print("Failed to get image: %s", url)
        return None
    if image.mode != "RGB":
        print("Convert image mode to RGB: %s", url)
        image = image.convert("RGB")
    return image


def dummy_business_card(i: int) -> BusinessCard | None:
    image = fetch_image_from_url(PICSUM_URL)
    if image is None:
        return None

    mean_color = np.mean(np.array(image), axis=(0, 1)).astype(int)

    text_color = color_invert(*mean_color)
    draw = ImageDraw.Draw(image)

    faker = Faker("ja_JP")

    text_x = randint(50, 100)

    # 左上の適当な位置とサイズを選び、会社名を書く
    company_point_x, company_point_y, company_size = (
        text_x,
        randint(50, 100),
        randint(20, 30),
    )
    company_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        company_size,
    )
    company = faker.company()
    draw.text(
        (company_point_x, company_point_y),
        company,
        font=company_font,
        fill=text_color,
    )
    company_bounding_box = draw.textbbox(
        (company_point_x, company_point_y),
        company,
        font=company_font,
    )

    # 会社名の下に適当な位置とサイズで名前を書く
    name_point_x, name_point_y, name_size = (
        text_x,
        company_bounding_box[3] + randint(10, 20),
        randint(30, 50),
    )
    name_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        name_size,
    )
    name = faker.name()
    draw.text(
        (name_point_x, name_point_y),
        name,
        font=name_font,
        fill=text_color,
    )

    detail_font_size = 20
    # 左下の適当な位置とサイズでメールアドレスを書く
    email_point_x, email_point_y = (
        text_x,
        randint(320, 360),
    )
    email_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    email = faker.email()
    draw.text(
        (email_point_x, email_point_y),
        email,
        font=email_font,
        fill=text_color,
    )
    email_bounding_box = draw.textbbox(
        (email_point_x, email_point_y),
        email,
        font=email_font,
    )

    # メールアドレスの下に電話番号を書く
    phone_point_x, phone_point_y = (
        text_x,
        email_bounding_box[3] + 5,
    )
    phone_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    phone = faker.phone_number()
    draw.text(
        (phone_point_x, phone_point_y),
        phone,
        font=phone_font,
        fill=text_color,
    )
    phone_bounding_box = draw.textbbox(
        (phone_point_x, phone_point_y),
        phone,
        font=phone_font,
    )

    # 電話番号の下に会社URLを書く
    url_point_x, url_point_y = (
        text_x,
        phone_bounding_box[3] + 5,
    )
    url_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    url = faker.url()
    draw.text(
        (url_point_x, url_point_y),
        url,
        font=url_font,
        fill=text_color,
    )
    url_bounding_box = draw.textbbox(
        (url_point_x, url_point_y),
        url,
        font=url_font,
    )

    # 会社URLの下に住所を書く
    address_point_x, address_point_y = (
        text_x,
        url_bounding_box[3] + 5,
    )
    address_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    address = faker.address()
    draw.text(
        (address_point_x, address_point_y),
        address,
        font=address_font,
        fill=text_color,
    )

    # 画像を保存
    image.save(f"{IMAGE_DIR}/{i}.png")

    return BusinessCard(
        image_path=f"{IMAGE_DIR}/{i}.png",
        company=company,
        name=name,
        email=email,
        phone_number=phone,
        address=address,
    )


if __name__ == "__main__":
    cards = []

    i = 0
    while i < DATASET_LENGTH:
        card = dummy_business_card(i)

        if card is None:
            continue

        print("Generated: %s", card)
        cards.append(asdict(card))

        i += 1

    print("Generated %s cards", len(cards))

    with LABEL_PATH.open("w") as f:
        f.write(json.dumps(cards, ensure_ascii=False, indent=4))
