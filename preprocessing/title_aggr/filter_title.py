import re
import json
import pandas as pd

class filter_title(object):
    def __init__(self):
        # 加载词典
        with open('../conf/replace_dict_1.json', 'r') as f:
            self.replace_dict_1 = json.load(f)
        with open('../conf/replace_dict_2.json', 'r') as f:
            self.replace_dict_2 = json.load(f)

        ## 加载 FUN 和 RES 表
        with open('../conf/RES.txt', 'r') as f:
            res_data = f.read().split('\n')
        with open('../conf/FUN.txt', 'r') as f:
            fun_data = f.read().split('\n')

        self.keywordsList = set(fun_data + res_data)

        ## 加载title, company库
        set2title_df = pd.read_csv('../conf/set2title_df.csv')
        self.set2title_dict = set2title_df.set_index('set')['title'].to_dict()
        self.set_titleList = set(self.set2title_dict.keys())


    def split_title(self, title):
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        title_split = re.split(pattern, title)
        while '' in title_split:
            title_split.remove('')
        return title_split


    def get_alpha_str(self, s):
        result = re.split(r'[^A-Za-z]', s)
        while '' in result:
            result.remove('')

        result = ' '.join(result)
        return result


    def filter(self, title):
        '''
        input: 原文本中抽取的title
        output: 规范化后的title转小写
        '''

        # 删除括号内的东西
        if '(' and ')' in title:
            title = re.sub(u" \\(.*?\\)|\\[.*?]|\\{.*?}", "", title)

        # 删除非英文片段
        title = self.get_alpha_str(title)

        # 匹配规范字典
        for word in self.replace_dict_1.keys():
            if word in title:
                title = title.replace(word, self.replace_dict_1[word])

        title = title.lower()
        # print(self.split_title(title))
        title = ' '.join([self.replace_dict_2[word] if word in self.replace_dict_2.keys() else word for word in self.split_title(title)])

        return title


    def standardize(self, title):
        # 从title中抽取keywords, 并且进行除重排序去匹配title库
        # 若能成功匹配，返回一个标准化的title, 若不能则返回False

        key = []
        word_list = self.split_title(self.filter(title))
        #    print(word_list)
        for word in word_list:
            if word in self.keywordsList:
                key.append(word)
        #    print(key)
        key = str(sorted(set(key)))
        #    print(key)
        if key in self.set_titleList:
            return self.set2title_dict[key]
        else:
            return False