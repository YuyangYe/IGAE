#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import pickle
import os
import re
import pdb

YEAR_TEMPLATE=r"([0-9]{4})\s*年"

DEBUG = 0

large_dict = dict()
# large_dict[university][major][year] = count
YEARS = {2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019}

def read_universities(univlist_filename: str) -> set:
    univs = set()
    fd = open(univlist_filename, 'r')
    for line in fd:
        univs.add(str(line).strip('\n').rstrip())
    #if (DEBUG):
    #    print(univs)
    fd.close()
    return univs

def process_input(src_filename: str,
                  univs: set,
                  keywords: set):
    global large_dict
    fd_src = open(src_filename, 'r')
    re_pattern = re.compile(YEAR_TEMPLATE)
    for idx, line in enumerate(fd_src):
        line_break1 = line.split(';')
        try:
            uid = line_break1[0]
            if (DEBUG):
                print(uid, end=' ')
            if len(uid) == 0:
                continue
            past_educations = line_break1[1].split('.')
            if len(past_educations) == 0:
                continue
            curr_univs = list()
            ignore = False
            for edu in past_educations:
                edu = edu.split(',')
                univ_name = edu[2]
                major = edu[3].lower()
                end_year = edu[5]
                end_year = re_pattern.search(end_year)
                start_year = edu[4]
                start_year = re_pattern.search(start_year)
                if end_year is None or start_year is None:
                    continue
                start_year = int(start_year.group(1))
                end_year = int(end_year.group(1))
                univ_name = str(univ_name)
                if (DEBUG):
                    print(univ_name, end='!')
                if univ_name not in univs:
                    continue
                if univ_name not in large_dict:
                    large_dict[univ_name] = dict()
                for keyword in keywords:
                    if major.find(keyword) != -1:
                        if keyword not in large_dict[univ_name]:
                            large_dict[univ_name][keyword] = dict()
                        for year in range(start_year, end_year+1):
                            if year not in YEARS:
                                continue
                            if year not in large_dict[univ_name][keyword]:
                                large_dict[univ_name][keyword][year] = 0
                            large_dict[univ_name][keyword][year] += 1
        except KeyboardInterrupt:
            raise
    fd_src.close()


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Invalid argument, expect <src_dirname> <university_list_filename> <keyword_name>")
        return 1
    src_dir = sys.argv[1]
    univ_list_filename = sys.argv[2]
    keywords_filename = sys.argv[3]
    keywords = set()
    with open(keywords_filename, 'r', encoding='utf-8') as fd:
        keywords = set(fd.read().lower().split(','))
    univs = read_universities(univ_list_filename)
    for file in os.listdir(src_dir):
        src_filename = f = os.path.join(src_dir, file)
        if not os.path.isfile(f):
            continue
        process_input(src_filename, univs, keywords)
    print(large_dict)

if __name__=="__main__":
    main()