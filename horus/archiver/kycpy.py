import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import pandas as pd
from io import StringIO
import os
pytesseract.pytesseract.tesseract_cmd = r"media\Tesseract-OCR\tesseract.exe"

def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed

set_pandas_options()

def choice(img,df1,df2,df3):
    template = cv2.imread('media/temp/res_temp.jpg',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    template1 = cv2.imread('media/temp/visa_temp.jpg',0)
    w, h = template1.shape[::-1]
    res = cv2.matchTemplate(img,template1,cv2.TM_CCOEFF_NORMED)
    threshold1 = 0.7
    loc1 = np.where( res >= threshold1)
    #print(loc)
    #print(loc1)
    if (len(loc[0])>0):
        df1=res_id_func(img,df1)
    elif(len(loc1[0])>0):
         df2=visa_main_func(img,df2)
    else:
        df3=pass_main_func(img,df3)
    return df1,df2,df3

def visa_main_func(img,df):
    l=int(len(img)/2)
    if len(img)>620:
        img1=img[l:]
    else:
        img1=img
    text=pytesseract.image_to_string(img1, lang="eng",config="--psm 6")
    txt=text.splitlines()
    new_list = [elem for elem in txt if elem.strip()]
    for i in range(len(new_list)):
        o=re.sub(r"[^a-z\ ]","",new_list[i].lower())
        if (" uld " in o) or (" uid " in o):
            #new_list[i].lower()
            a=new_list[i-1]
        if "name" in o:
            b=new_list[i-1]
        if "passport no" in o:
            c=new_list[i-1]
    l=re.sub(r"[^\d{4}\/\d{1,2}\/\d{1,2}]"," ",text).split()
    li=[]
    for ele in l:
        if((len(ele.split("/"))==3) & ((len(ele.split("/")[-1])==2))):
            li.append(ele)
    expire=li[0]
    expiry=expire[8:10]+"/"+expire[5:7]+"/"+expire[0:4]
    issue=li[1]
    issue_date=issue[8:10]+"/"+issue[5:7]+"/"+issue[0:4]
    uid=re.sub(r"[^0-9]","",a)
    name=re.sub(r"[^a-z,A-Z]"," ",b)
    passport=re.sub(r"[^0-9]","",c)
    df=df.append([[uid,name,passport,issue_date,expiry]],ignore_index=True)
    return df


def pass_main_func(img,df):
    flag=0
    l=int(len(img)/2)
    if len(img)>620:
        img1=img[l:]
    else:
        img1=img
    img2=cv2.resize(img1,None,fx=1.8,fy=1.5)
    text=pytesseract.image_to_string(img2, lang="eng",config="--psm 11")
    text=text.lower()
    txt=text.splitlines()
    new_list = [elem for elem in txt if elem.strip()]
    l=[]
    for i in reversed(new_list):
        if '<<' in i:
            i=i.replace(' ','')
            l.append(i)
            #print(l)
    nation=l[1].split("<")[1][0:3]
    #print(l[1].split("<<<<")[0])
#     name_ls = l[1].split("<<<<")[0].split("<")[1:]
#     name_ls[0] = name_ls[0][3:]
    name=(l[1].split("<<<<")[0].split("<<"))
    #l_name=" ".join(name[0].split("<")[1:])[3:]
    c=name[1].split("<")
    f_name=" ".join(c)
    l_name=" ".join(name[0].split("<")[1:])[3:]
    #print((l[1].split("<<<<")[0].split("<")[1:]))
    P_no=l[0].split("<")[0]
    if(nation in P_no):
        P_no=P_no.split(nation)[0][:-1]
    sex=l[0].split(nation)[1][7]
    exp_date=l[0].split(nation)[1][8:14]
    dob=l[0].split(nation)[1][0:6]
    dob=dob[4:6]+"/"+dob[2:4]+"/"+dob[0:2]
    exp_date=exp_date[4:6]+"/"+exp_date[2:4]+"/"+exp_date[0:2]
    df=df.append([[P_no,l_name,f_name,nation,dob,sex,exp_date]],ignore_index=True)
    return df        

def res_id_func(img,df):
    l=int(len(img)/2)
    if len(img)>620:
        img1=img[l:]
    else:
        img1=img
    img2=cv2.resize(img,None,fx=1.25,fy=1)
    k=np.ones((2,2),np.uint8)
    img2=cv2.dilate(img2,k)
    ret,thresh1 = cv2.threshold(img2,145,255,cv2.THRESH_BINARY)
    text=pytesseract.image_to_string(thresh1, lang="eng",config="--psm 6")
    text=text.lower()
    txt=text.splitlines()
    new_list = [elem for elem in txt if elem.strip()]
    l=[]
    for i in reversed(new_list):
        i=i.replace(' ','')
        l.append(i)
    l[0:3]
    name=" ".join(l[0].split("<<<<")[0].split("<"))
    l[1]=re.sub("<","",l[1])
    dob=l[1][0:6]
    gender=l[1][7]
    exp_date=l[1][8:14]
    nation=l[1][15:18]
    dob=dob[4:6]+"/"+dob[2:4]+"/"+dob[0:2]
    expiry=exp_date[4:6]+"/"+exp_date[2:4]+"/"+exp_date[0:2]
    l[2]=re.sub("<","",l[2])
    id_no=l[2][-15:]
    df=df.append([[id_no,name,nation,gender,dob,expiry]],ignore_index=True)
    return df


        
def main(inp):
    img=cv2.imread(inp,0)
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df3=pd.DataFrame()
    df1,df2,df3=choice(img,df1,df2,df3)
    a=len(df1)
    b=len(df2)
    c=len(df3)
    if a>b and a>c:
        df1.columns=["ID No","Name","Nation","gender","DOB","Expiry date"]

        return 1,df1
    elif b>a and b>c:
        df2.columns=["UID","Name","Passport Number","Issue date","Expiry date"]

        return 2,df2
    else:
        df3.columns=["Passport Number","last_name","first_name","Nationality","DOB","SEX","EXP_DATE"]
        return 3,df3