import requests
from bs4 import BeautifulSoup
import re
import csv


def analysis(url, area):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/124.0.0.0 \Safari/537.36'
    }

    content = requests.get(url=url, headers=headers).content
    soup = BeautifulSoup(content, 'lxml')
    divs = soup.select(".sellListContent li .info")

    with open(f"./Data/二手房信息{area}.csv", mode='a', encoding='utf-8', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['标题', '小区', '区域', '总价', '单价', '户型', '面积', '朝向',
                                                   '装修情况', '楼层类别', '总层数', '建筑类型'])

        # 如果文件为空，写入表头
        if f.tell() == 0:
            csv_writer.writeheader()

        for div in divs:
            title_selects = div.select('.title a')  # 标题
            title = title_selects[0].text if title_selects else ""

            # 区域划分 0为小区，1为地区
            areas_selects = div.select('.positionInfo a')  # 区域
            area0 = areas_selects[0].text if areas_selects else ""
            area1 = areas_selects[1].text if len(areas_selects) > 1 else ""

            totalPrice = div.select('.sellListContent li .info .priceInfo .totalPrice')
            total_price = totalPrice[0].text if totalPrice else ""

            unitPrice = div.select('.sellListContent li .info .priceInfo .unitPrice')
            unit_price = unitPrice[0].text if unitPrice else ""

            # 房屋信息，具体划分
            houseInfo = div.select('.houseInfo')
            house_info = houseInfo[0].text.split("|") if houseInfo else [""] * 6

            houseType = house_info[0]  # 户型
            houseArea = house_info[1]  # 面积
            houseFaceTo = house_info[2]  # 朝向
            houseFurnish = house_info[3]  # 装修情况

            houseFloor = house_info[4]  # 楼层
            match = re.match(r'(.+)\((共\d+层)\)', houseFloor)  # 使用正则表达式匹配楼层信息
            if match:
                floor_describe = match.group(1)  # 楼层
                total_floor = match.group(2)  # 总层数
            else:
                floor_describe = ""
                total_floor = ""

            houseBuild = house_info[5]  # 建筑类型

            house_dict = {
                '标题': title,
                '小区': area0,
                '区域': area1,
                '总价': total_price,
                '单价': unit_price,
                '户型': houseType,
                '面积': houseArea,
                '朝向': houseFaceTo,
                '装修情况': houseFurnish,

                # '楼层': houseFloor,
                '楼层类别': floor_describe,
                '总层数': total_floor,

                '建筑类型': houseBuild,
            }

            csv_writer.writerow(house_dict)
        print(f"{url}解析成功({area})")

