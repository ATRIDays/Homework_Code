def switch_ch(pinyin):
    switch_dict = {
    'binhu': '滨湖区',
    'liangxi': '梁溪区',
    'xinwu': '新吴区',
    'huishan': '惠山区',
    'xishan': '锡山区',
    'jiangyinshi': '江阴市',
    'yixingshi': '宜兴市'
    }
    return switch_dict.get(pinyin, "未找到对应的汉字")
