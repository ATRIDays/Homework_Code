from URL_Analysis import analysis

root_url = 'https://wx.lianjia.com/ershoufang/'
areas = ['binhu', 'liangxi', 'xinwu', 'huishan', 'xishan', 'jiangyinshi', 'yixingshi']
index = 100
for area in areas:
    for i in range(index):
        final_url = f"{root_url}{area}/pg{i+1}"
        # print(final_url)
        analysis(final_url, area)