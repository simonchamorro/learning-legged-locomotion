
import os
import time
from adafruit_extended_bus import ExtendedI2C as I2C
i2c = I2C(4)
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont # requires that pillow is installed
import adafruit_ssd1306
import urllib.request

WIDTH = 128
HEIGHT = 64  # Change to 64 if needed
BORDER = 5

OK = 'OK'
NOT_OK = ''
LOOP_TIME = 10


def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host, timeout=0.3)
        return True
    except:
        return False

last_loop = time.time()
while True:
    now = time.time()
    if now - last_loop < LOOP_TIME:
        continue
    last_loop = time.time()
    oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)
    oled.fill(0)
    oled.show()
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw Status Text
    wifi_ok = OK if connect() else NOT_OK
    robot_ok = OK if os.system('systemctl is-active --quiet robot') == 0 else NOT_OK
    joy_ok = OK if os.system('systemctl is-active --quiet joystick') == 0 else NOT_OK
    text_wifi = "Wifi " + wifi_ok 
    text_robot = "Robot " + robot_ok 
    text_joy = "Joy " + joy_ok 
    text = text_wifi + '\n' + text_robot + '\n' + text_joy
    (font_width, font_height) = font.getsize(text)
    draw.text(
        (0, 0),
        text,
        font=font,
        fill=255,
    )

    # Display image
    oled.image(image)
    oled.show()