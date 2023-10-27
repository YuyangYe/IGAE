#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A tool to convert work_experience Linkedin dataset to company transition pairs

For example, a person join company A first, then B,
A,B will be written to the result data file

Tianyi Cui, 2023, All Rights Reserved
"""
DEBUG=False

import sys

companies = set()

def show_help():
    print("%s <src_filename> <dst_filename> <companylist_filename>" % sys.argv[0])

def read_companies(companylist_filename: str):
    fd = open(companylist_filename, 'r')
    for line in fd:
        companies.add(str(line).strip('\n'))
    if (DEBUG):
        print(companies)
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
                company_name = experience.split(',')[3]
                company_name_wordlist = company_name.split()
                if (DEBUG):
                    print(company_name_wordlist, " ", end="")
                keyword = set(company_name_wordlist) & companies
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
    companylist_filename = sys.argv[3]
    read_companies(companylist_filename)
    process_input(src_filename, dst_filename)


if __name__ == "__main__":
    main()