###################################################
company.tar.bz2:包含公司数据集

包含内容如下：
0|-- companyName:	名称
1|-- industry:	领域
2|-- size:	规模
3|-- description:	描述
4|-- specialties:	专注领域
5|-- companyType:	类型
6|-- yearFounded:	成立时间
7|-- location:	地点
8|-- website:	网址

存储格式如下：用逗号分隔
每行对应一个公司的信息：
0;1;2;3;4;5;6;7;8
###################################################

###################################################
profile.tar.bz2:包含个人profile

包含内容如下：
 0|-- url: string (nullable = true)，用户id
 1|-- name: string (nullable = true)，用户名称
 2|-- title: string (nullable = true)，当前岗位
 3|-- address: string (nullable = true)，地址
 4|-- org_url: string (nullable = true)，公司链接
 5|-- org_name: string (nullable = true)，公司名称
 6|-- industry: string (nullable = true)，所属领域

存储格式如下：
每行对应一个人的profile，id之后用";"隔开，字段之间用","隔开。如下
0; 1,2,3,4,5,6
###################################################

###################################################
work_experience.tar.bz2:包含个人工作经历信息

包含内容如下：
 0|-- id: string (nullable = true)，用户id
 1|-- idx: long (nullable = true)，当前经历在所有经历中的排序，从0开始计数
 2|-- position: string (nullable = true)，岗位
 3|-- comp_url: string (nullable = true)，公司url
 4|— comp_name: string (nullable = true)，公司名称
 5|— start_date: string (nullable = true)，开始日期
 6|-- end_date: string (nullable = true)，结束时期
 7|-- duration: string (nullable = true)，经历长度
 8|-- city: string (nullable = true)，公司地点
 9|-- desc: string (nullable = true)，岗位描述

存储格式如下：
每行对应一个人的简历，用最左侧的数字代表对应的字段：id之后用";"隔开，经历之间用"."隔开，经历的字段之间用","隔开。如下
0; 1,2,3,4,5,6,7,8,9. 1,2,3,4,5,6,7,8,9. ...
###################################################

###################################################
education_experience.tar.bz2:包含个人教育经历信息

包含内容如下：
 0|-- id: string (nullable = true)，用户id
 1|-- idx: long (nullable = true)，当前经历在所有经历中的排序，从0开始计数
 2|-- sch_url: string (nullable = true)，学校链接
 3|-- sch_name: string (nullable = true)，学校名称
 4|-- subject: string (nullable = true)，学历信息，比如，本科，硕士等
 5|-- start_date: string (nullable = true)，开始时间
 6|-- end_date: string (nullable = true)，结束时间

存储格式如下：
每行对应一个人的教育经历，用最左侧的数字代表对应的字段：id之后用";"隔开，经历之间用"."隔开，经历的字段之间用","隔开。如下
0; 1,2,3,4,5,6. 1,2,3,4,5,6. ...
###################################################

###################################################
project.tar.bz2:包含个人项目经历

包含内容如下：
 0|-- id: string (nullable = true)，用户id
 1|-- idx: long (nullable = true)，当前项目在所有项目中的排序，从0开始计数
 2|-- title: string (nullable = true)，项目名称
 3|-- start_date: string (nullable = true)，开始时间
 4|-- end_date: string (nullable = true)，结束时间
 5|-- desc: string (nullable = true)，描述
 6|-- mem_str: string (nullable = true)，成员

 存储格式如下：
每行对应一个人的项目经历，用最左侧的数字代表对应的字段：id之后用";"隔开，项目经历之间用"."隔开，经历的字段之间用","隔开。如下
0; 1,2,3,4,5,6. 1,2,3,4,5,6. ...
###################################################

###################################################
skill.tar.bz2:包含个人技能

包含内容如下:
 |-- id: string (nullable = true)
 |-- skills: string (nullable = true)

 存储格式如下：
id;skills
###################################################










