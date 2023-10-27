#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A tool to convert work_experience Linkedin dataset to university transition pairs

For example, a person join university A first, then B,
A,B will be written to the result data file

Tianyi Cui, 2023, All Rights Reserved
"""
DEBUG=False

import sys

universities = set()

def show_help():
    print("%s <src_filename> <dst_filename> <universitylist_filename>" % sys.argv[0])

def read_universities(universitylist_filename: str):
    fd = open(universitylist_filename, 'r')
    for line in fd:
        universities.add(str(line).strip('\n'))
    if (DEBUG):
        print(universities)
    fd.close()

def process_input(src_filename: str,
                  dst_filename: str):
    fd_src = open(src_filename, 'r')
    fd_dst = open(dst_filename, 'a')
    for idx, line in enumerate(fd_src):
        line_break1 = line.split(';')
        try:
            uid = line_break1[0]
            if len(uid) == 0:
                continue
            past_experiences = line_break1[1].split('.')
            last_job = ""
            for experience in past_experiences:
                university_name = experience.split(',')[3]
                university_name_wordlist = university_name.split()
                if (DEBUG):
                    print(university_name_wordlist, " ", end="")
                keyword = set(university_name_wordlist) & universities
                if len(keyword) > 0:
                    true_name = next(iter(keyword))
                    if last_job != "":
                        fd_dst.write("%s %s\n" % (last_job, true_name))
                        fd_dst.flush()
                    last_job = true_name
            if (DEBUG):
                print("")
        except KeyboardInterrupt:
            raise
        except:
            pass
    fd_dst.close()
    fd_src.close()

def main():
    if len(sys.argv) != 4:
        show_help()
        return
    src_filename = sys.argv[1]
    dst_filename = sys.argv[2]
    universitylist_filename = sys.argv[3]
    read_universities(universitylist_filename)
    process_input(src_filename, dst_filename)


if __name__ == "__main__":
    main()