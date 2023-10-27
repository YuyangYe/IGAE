# University ranking project
This repository contains the code for preprocessing the data from LinkedIN.
# Usage
## Filter out users whose education are done by target universities
./filter_uid_by_edu.py <src folder containing user education info> <dst output pickle> <target university list>
## Add working experience to that pickle file
./unpack_pickle_add_work.py <src folder containing user working info> <src pickle file from previous step> <dst pickle>
## Generate company transition edges list from user working experiences
./get_pair.py <src_company_file> <dst_company_file> <company_list>
## Generate university transition edges list from user education experiences
./get_pair.py <src_university_file> <dst_university_file> <university_list>