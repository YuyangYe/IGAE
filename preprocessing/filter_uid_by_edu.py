#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import pickle
import os
import re

YEAR_TEMPLATE=r"([0-9]{4})\s*年"

DEBUG = 0

large_dict = dict()
year_count = 0

# uid -> ([(school name, end)],[])

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
                  univs: set):
    global large_dict
    global year_count
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
                end_year = edu[5]
                end_year = re_pattern.search(end_year)
                if end_year is not None:
                    end_year = int(end_year.group(1))
                    year_count += 1
                univ_name = str(univ_name)
                if (DEBUG):
                    print(univ_name, end='!')
                if univ_name not in univs:
                    ignore = True
                    if not DEBUG:
                        break
                    else:
                        print("Not match!!!%s!!!" % univ_name)
                curr_univs.append((copy.deepcopy(univ_name), end_year))
            if (DEBUG):
                print('')
            if not ignore:
                large_dict[uid] = (curr_univs,None)
        except KeyboardInterrupt:
            raise
    fd_src.close()


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Invalid argument, expect <src_filename> <dst_filename> <university_list_filename>")
        return 1
    src_dir = sys.argv[1]
    dst_filename = sys.argv[2]
    univ_list_filename = sys.argv[3]
    univs = read_universities(univ_list_filename)
    for file in os.listdir(src_dir):
        src_filename = f = os.path.join(src_dir, file)
        if not os.path.isfile(f):
            continue
        print("current %d record, duration %d, start process %s" % (len(large_dict), year_count, src_filename))
        process_input(src_filename, univs)
    out_fd = open(dst_filename, 'wb')
    pickle.dump(large_dict, out_fd)
    out_fd.close()

if __name__=="__main__":
    main()