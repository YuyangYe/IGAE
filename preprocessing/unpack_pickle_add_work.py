#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <work dir> <in_pickle> <out pickle>

import sys
import pickle
import os
import re
import copy

RE_TEMPLATE=r"^(\d{4})/(\d+)$"

large_dict = dict()
large_dict_out = dict()
global_matched_cnt = 0
global_end_date_cnt = 0
DEBUG = 0

def process_input(src_filename: str):
    global large_dict
    global global_matched_cnt
    global global_end_date_cnt
    global large_dict_out
    fd_src = open(src_filename, 'r')
    end_date_template = re.compile(RE_TEMPLATE)
    for idx, line in enumerate(fd_src):
        line_break1 = line.split(';')
        try:
            uid = line_break1[0]
            if len(uid) == 0:
                continue
            if not uid in large_dict:
                continue
            past_experiences = line_break1[1].split('.')
            companies = []
            for experience in past_experiences:
                experience = experience.split(',')
                company_name = experience[3]
                end_date = experience[5]
                end_date = end_date_template.match(end_date)
                if end_date is not None:
                    year = int(end_date.group(1))
                    month = int(end_date.group(2))
                    if month <= 6:
                        month = 0
                    else:
                        month = 1
                    end_date = year * 10 + month
                    global_end_date_cnt += 1
                companies.append((company_name,end_date))
            large_dict_out[uid] = list()
            large_dict_out[uid] = [large_dict[uid][0], companies]
        except KeyboardInterrupt:
            raise
        global_matched_cnt += 1
    fd_src.close()

def main():
    global large_dict
    if len(sys.argv) != 4:
        print("Invalid parameter")
        exit(1)
    work_dir = sys.argv[1]
    in_pickle_name = sys.argv[2]
    out_pickle_name = sys.argv[3]
    in_fd = open(in_pickle_name, "rb")
    large_dict = pickle.load(in_fd)
    in_fd.close()
    print("User list loaded")
    for file in os.listdir(work_dir):
        src_filename = f = os.path.join(work_dir, file)
        if not os.path.isfile(f):
            continue
        print("start process %s, matched %d, end_date calculated %d" % (src_filename, global_matched_cnt, global_end_date_cnt))
        process_input(src_filename)
    out_fd = open(out_pickle_name, "wb")
    pickle.dump(large_dict_out, out_fd)
    out_fd.close()

if __name__=="__main__":
    main()