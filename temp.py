import re
import requests
from bs4 import BeautifulSoup as bs

def get_weather_data(city):
    html = requests.get(f'https://search.naver.com/search.naver?query={city}날씨')
    if html.status_code != 200:
        return None

    bs_object = bs(html.text, 'html.parser')
    address = bs_object.select('div.title_area._area_panel > h2.title')[0].text
    temp = bs_object.select('.temperature_text > strong')[0].text
    weath = bs_object.select('div.temperature_info > p > span.weather.before_slash')[0].text
    r1 = re.compile('[-0-9]+')
    temp = int(r1.findall(temp)[0])

    total = bs_object.select('div.report_card_wrap > ul > li > a > span')[:4]
    additional_info = {text: tag.text for text, tag in zip(['미세먼지', '초미세먼지', '자외선', '일몰'], total)}

    return {
        '위치': address,
        '온도': temp,
        '날씨': weath,
        '부가정보': additional_info
    }
