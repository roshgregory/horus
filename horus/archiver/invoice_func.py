
from string import punctuation
from datetime import datetime
import urllib
import datetime
import numpy as np
import pytesseract
import re
import os
import configparser
import pandas as pd
from datetime import datetime
from pdf2jpg import pdf2jpg
import shutil
from cv2 import cvtColor, findContours, getStructuringElement, morphologyEx, threshold
import cv2
import os
import json
import difflib

number_of_success=0

import datetime as dt
import xlrd
from openpyxl import load_workbook

# config file reader
###ADDDDDDD

##ADD 
#audit_log=r'audit_log.xlsx'


#BASE_DIR = r'D:\work\retail al\ui'
#SUCCESS_folder=r'D:\work\retail al\outputs\success'
#csvlocation=r'D:\work\retail al\ui\output'
#templocation=r'D:\work\retail al\ui\temp'
#pdf=os.path.join(BASE_DIR, r"media")+"\PDF"
#failed=r'D:\work\retail al\outputs\fail'
#con=85
#pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#lan='eng_best'


config = configparser.RawConfigParser()
config.read(r'ConfigFile.properties')

pytesseract.pytesseract.tesseract_cmd = (config.get('Tesseract', 'pytesseract.pytesseract.tesseract_cmd'))

lan = (config.get('language', 'lang'))




def main_func(Original_images,  n ,fileName,fi):
    start=0
    x = dt.datetime.now() 
    audit_start_time=x.strftime("%X")



    companylist=[]
    invoiceNumberlist=[]
    invoiceDatelist=[]
    remarkslist=[]
    amountlist=[]

    companylistconf=[]
    invoiceNumberlistconf=[]
    invoiceDatelistconf=[]
    remarkslistconf=[]
    amountlistconf=[]

    df=pd.DataFrame()
    flag=0
    date_flag=0
    total='-'
    remark='-'
    for i in range(len(Original_images)):
        Original_img=White(Original_images[i])
        img_gr=cv2.cvtColor(Original_img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img_gr,None,fx=1.25,fy=1.3,interpolation=cv2.INTER_LINEAR)

        # String1 is used for confidence validation
        string = pytesseract.image_to_string(img,lang='eng_best',config='--psm 6 -c preserve_interword_spaces=1')
        string1 = pytesseract.image_to_data(img,lang='eng_best',config='--psm 6 -c preserve_interword_spaces=1')

        string = string.replace('Totai',' Total ').replace('Tatal',' Total ').replace('Tetal',' Total ')
        string = string.replace('Fatal',' Total ').replace('Fotal',' Total ').replace('Tatat',' Total ')
        string = string.replace('Tata!',' Total ').replace('Folat',' Total ').replace('Tolal',' Total ')
        string = string.replace('Toral',' Total ').replace('Totat',' Total ').replace('Taral','Total')

        string = string.replace('Tata ','Total')
        string = string.replace('Tota ',' Total ')
        string = string.replace('Tota? :',' Total ')
        string = string.replace('Totaé:',' Total ')
        #print('string--1',string)

        lines = string1.split("\n")
        column = lines.pop(0)
        linepd1 = pd.DataFrame([x.split('\t') for x in lines])
        linepd1.columns = ['level','page_num', 'block_num', 'par_num', 'line_num', 'word_num','left','top','width','height','conf','text']
        linepd1 = linepd1[linepd1.text != '']
        AEDline = linepd1[linepd1.text.str.contains('AED')]['line_num']

        break_words=["Remarks","Invoice Date","Invoice Number","Company","AED","Invoice","Number","Date"]
        break_word=[word for word in break_words if word in string]
        if("AED" in string.split(break_word[0])[1].splitlines()[1]):
            f=string.split(break_word[0])[1].splitlines()[2:]
        else:
            f = string.split(break_word[0])[1].splitlines()[1:]
            AEDline = linepd1[linepd1.text.str.contains(break_word[0])]['line_num']

        print('Processing Page:',i)
        for j in range(len(f)):
            #print(f[j])
            rawstr = cleanRawStr(f[j])

            #print('rawstr---',rawstr)
            if len(rawstr.strip()) > 0:
                listf = [i.lstrip() for i in rawstr.split('  ')]
#mine			print(listf)
                listf = cleanlist(listf)
                print("PRITNING STATEMENT")
                print(listf)
                dfconfidence = linepd1[linepd1['line_num'] == str(j+int(AEDline.get_values()[0])+1)]
                listf = [i for i in listf if len(i) > 1]

                #added to remove
                listf = [i.lstrip(punctuation) for i in listf]
                if listf:
                    listf[0] = listf[0].strip(punctuation).strip()

                    if listf[0].endswith('Tota') or listf[0].endswith('Tata'):
                        a = listf[0]
                        listf[0] = a[:-4]
                        listf.insert(1, 'Total')
                    if len(listf)==3 and (listf[1]=='Tata' or listf[1]=='Tota'):
                        listf[1]='Total'
                    if len(listf) == 1:
                        new=match_func(listf[0])
                        companylist.append(new.lstrip(punctuation))
                        companylistconf.append(findconfoflist(listf[0], dfconfidence))

                        invoiceNumberlist.append('')
                        invoiceDatelist.append('')
                        remarkslist.append('')
                        amountlist.append('')

                        invoiceNumberlistconf.append(False)
                        invoiceDatelistconf.append(False)
                        remarkslistconf.append(False)
                        amountlistconf.append(False)
                    elif len(listf) == 3:
                        if 'Total' in listf or 'total' in listf or 'tatal' in listf:
                                remarkslist.append(listf[0])
                                remarkslistconf.append(findconfoflist(listf[0], dfconfidence))

                                amountlist.append(listf[-1])
                                amtconfVal = findconfoflist(listf[-1], dfconfidence)
                                amtconfVal = checkAmountConf(listf[-1], amtconfVal)
                                amountlistconf.append(amtconfVal)

                                companylist.append('')
                                companylistconf.append(False)

                                invoiceNumberlist.append('')
                                invoiceNumberlistconf.append(False)

                                invoiceDatelist.append('Total')
                                invoiceDatelistconf.append(False)
                        else:
                                print(listf)
                                if len(listf[0].split())>1:
                                    invoiceNumberlist.append(''.join(i for i in listf[0].split()))
                                else:
                                    invoiceNumberlist.append(listf[0])
                                #need to check
                                invoiceNumberlistconf.append(findconfoflist(listf[0],dfconfidence))

                                if len(listf[1].split()) >1:
                                    inv_date=(listf[1].lstrip(punctuation)).split()[0].lstrip(punctuation)
                                    invoiceDatelist.append(inv_date)
                                    invoiceDatelistconf.append(invoiceDateConfCheck(inv_date,dfconfidence))

                                    remarktext=' '.join((listf[1].lstrip(punctuation)).split()[1:])
                                    remarkslist.append(remarktext.lstrip(punctuation))
                                    remarkslistconf.append(findconfoflist(remarktext.lstrip(punctuation),dfconfidence))
                                else:
                                    invoiceDatelist.append(listf[1].lstrip(punctuation))
                                    invoiceDatelistconf.append(invoiceDateConfCheck(listf[1].lstrip(punctuation),dfconfidence))

                                    remarkslist.append('')
                                    if listf[-1].isdigit() and not ('.' in listf[-1] or ',' in listf[-1]):
                                        remarkslistconf.append(True)
                                    else:
                                        remarkslistconf.append(False)

                                amountlist.append(listf[-1])
                                amtconfVal = findconfoflist(listf[-1],dfconfidence)
                                amtconfVal = checkAmountConf(listf[-1],amtconfVal)
                                amountlistconf.append(amtconfVal)

                                companylist.append('')
                                companylistconf.append(False)
                    elif len(listf) == 4:
                            if len(listf[0].split()) >1:
                                invoiceNumberlist.append(''.join(i for i in listf[0].split()))
                            else:
                                invoiceNumberlist.append(listf[0])
                            invoiceNumberlistconf.append(findconfoflist(listf[0],dfconfidence))

                            #Need to check
                            rem = ''
                            if len(listf[1].split())>1:
                                invoiceDatelist.append((listf[1].lstrip(punctuation)).split()[0].lstrip(punctuation))
                                rem = rem+' '+"".join(str(listf[1]).split()[1:])
                            else:
                                invoiceDatelist.append(listf[1].lstrip(punctuation))
                            invoiceDatelistconf.append(invoiceDateConfCheck(listf[1],dfconfidence))

                            remarkslist.append(rem+listf[-2])
                            #remarkslistconf.append(findconfoflist(listf[2],dfconfidence))
                            remarkslistconf.append(findconfoflist(rem+listf[-2],dfconfidence))

                            companylist.append('')
                            companylistconf.append(False)

                            amountlist.append(listf[-1])
                            # amountlistconf.append(findconfoflist(listf[-1],dfconfidence))
                            amtconfVal = findconfoflist(listf[-1], dfconfidence)
                            amtconfVal = checkAmountConf(listf[-1], amtconfVal)
                            amountlistconf.append(amtconfVal)
                    else:
                        if len(listf) == 2 and listf[0].endswith('Total'):
                            list_1 = listf[0].rsplit(' ', 1)

                            amountlist.append(listf[-1])
                            #amountlistconf.append(findconfoflist(listf[-1],dfconfidence))
                            amtconfVal= findconfoflist(listf[-1],dfconfidence)
                            amtconfVal=checkAmountConf(listf[-1],amtconfVal)
                            amountlistconf.append(amtconfVal)

                            remarkslist.append(list_1[0])
                            remarkslistconf.append(findconfoflist(list_1[0],dfconfidence))

                            companylist.append('')
                            companylistconf.append(False)

                            invoiceNumberlist.append('')
                            invoiceNumberlistconf.append(False)

                            invoiceDatelist.append('Total')
                            invoiceDatelistconf.append(False)
                        else:
                            if len(listf) == 2 and listf[-1].endswith('Total'):
                                amountlist.append('')
                                amountlistconf.append(True)

                                remarkslist.append(listf[0])
                                remarkslistconf.append(findconfoflist(listf[0],dfconfidence))

                                companylist.append('')
                                companylistconf.append(False)


                                invoiceDatelist.append('Total')
                                invoiceDatelistconf.append(False)
                                invoiceNumberlist.append('')
                                invoiceNumberlistconf.append(False)
                            elif len(listf) == 2 and '-' in listf[-1] :
                                amountlist.append('')
                                amountlistconf.append(True)

                                remarkslist.append('')
                                remarkslistconf.append(False)

                                companylist.append('')
                                companylistconf.append(False)


                                invoiceDatelist.append(listf[-1])
                                invoiceDatelistconf.append(findconfoflist(listf[-1],dfconfidence))

                                invoiceNumberlist.append(listf[0])
                                invoiceNumberlistconf.append(findconfoflist(listf[0],dfconfidence))
                            elif len(listf) == 2 and '-' in listf[0] and (',' in listf[-1] or '.' in listf[-1]):
                                amountlist.append(listf[-1])
                                amountlistconf.append(findconfoflist(listf[-1], dfconfidence))

                                invoiceDatelist.append(listf[0])
                                invoiceDatelistconf.append(findconfoflist(listf[0],dfconfidence))

                                remarkslist.append('')
                                remarkslistconf.append(False)

                                companylist.append('')
                                companylistconf.append(False)

                                invoiceNumberlist.append('')
                                invoiceNumberlistconf.append(False)
                            elif len(listf) == 5 :
                                if '-' in listf[1]:
                                    companylist.append('')
                                    companylistconf.append(False)

                                    invoiceNumberlist.append(listf[0])
                                    invoiceNumberlistconf.append(findconfoflist(listf[0],dfconfidence))

                                    invoiceDatelist.append(listf[1])
                                    invoiceDatelistconf.append(findconfoflist(listf[1],dfconfidence))

                                    remarkslist.append(listf[2])
                                    remarkslistconf.append(findconfoflist(listf[2],dfconfidence))

                                    amount =listf[3]+listf[4]
                                    amountlist.append(amount)
                                    amountlistconf.append(findconfoflist(amount, dfconfidence))
                                else:
                                    print('Skipped 5',listf)
                            else:
                                print('Skipped',listf)

    d = {    'Company'          : companylist,
             'Invoice Number'   : invoiceNumberlist,
             'Invoice Date'     : invoiceDatelist,
             'Remarks'          : remarkslist,
             'Amount paid(AED)' : amountlist,
             'Company_validate' : companylistconf,
             'invoice_validate' : invoiceNumberlistconf,
             'date_validate'    : invoiceDatelistconf,
             'remarks_validate' : remarkslistconf,
             'amt_validate'     : amountlistconf
        }
    df = pd.DataFrame(data=d)
    df = df[['Company', 'Invoice Number','Invoice Date','Remarks','Amount paid(AED)','Company_validate','invoice_validate','date_validate','remarks_validate','amt_validate']]
    df = df.reset_index(drop=True)
    df['Company_validate'] = np.where(df['Company'] == '', np.nan, df['Company_validate'].astype(str))
    df['Company'] = df['Company'].replace(r'^\s*$', np.nan, regex=True)
    col = ['Company','Company_validate']
    #dg= dg.replace(r'^\s*$', np.nan, regex=True)
    df.loc[:, col] = df.loc[:,col].ffill()
    df = df[~((df['Invoice Number'] == '')&(df['Invoice Date'] == '')&(df['Remarks'] == '')&(df['Amount paid(AED)'] == ''))]
    #df['Remarks'] = np.where(df['Invoice Date'] == 'Total', df['Company'], df['Remarks'])
    df.tail(1)['Company']=''

    df['Company'] = np.where(df['Invoice Date'] == 'Total', '', df['Company'])
    df.Company_validate=df.Company_validate.replace({ 'True':True,  'False':False})
    df['Company_validate'] = np.where(df['Company'] == '', False, df['Company_validate'])
    df = df.reset_index(drop=True)
        #validation
    sumtot=0
    start_count=0
    sub_amt_cnt = []
    for i in range(len(df)-1):
        amt_value=df["Amount paid(AED)"][i].lstrip(punctuation).lstrip().strip()
        print(amt_value)
        if len(amt_value)>=3:
            if amt_value[-1]=='-':
                amt_value='-'+amt_value[:-1]
#mine       if len(amt_value)>=3:
                amt_value = amt_value.rstrip('.')
            if amt_value[-3]==' ' or amt_value[-3]==',':
                amt_value="".join((amt_value[:-3],".",amt_value[-2:]))

        amt_value=amt_value.replace(' ','')
        amt_value=amt_value.replace(',','')
        # to remove punctuation
        amt_value=amt_value.replace('+','').replace(")",'').replace("(",'')
        pat=re.compile(r'[^0123456789.,\- ]+')
        amt_value=re.sub(pat, '', amt_value)
        pat1=re.compile(r'[..]+')
        amt_value=re.sub(pat1, '.', amt_value)
        amt_value=amt_value.replace(' ','').rstrip(punctuation)
        guess = "."
        occurrences = amt_value.count(guess)
        if occurrences>1:
            indices = [i for i, a in enumerate(amt_value) if a == guess]
            amt_value="".join((amt_value[:indices[-2]].replace('.',''),"",amt_value[indices[-2]+1:]))
        if amt_value.strip()=='':
            amt_value='0.00'
        df["Amount paid(AED)"][i]=amt_value
        if(df['Invoice Date'][i].strip(punctuation).strip()!='Total'):
            sumtot+=float(amt_value)
            sub_amt_cnt.append(i)
            if(start_count==0):
                    start=i
            start_count=1
        else:
            if(round(sumtot,2)==float(amt_value)):
                print('False')
                #df['amt_validate'][start:(i+1)]=False
            else:
                #print('True')
                df['amt_validate'][start:(i+1)]=True
            sumtot = 0
            start_count=0
            sub_amt_cnt=[]

    for index,row in df.iterrows():
        if(row['Invoice Date']=="Total"):
            df.ix[index,'Remarks']=match_func(str(row['Remarks']))
	
    df['Company_validate']='FALSE'
    df.loc[df['Invoice Date'] == 'Total', 'remarks_validate'] =False

    
    company_validate_count = (df['Company_validate']).sum()
    invoice_validate_count = (df['invoice_validate']).sum()
    date_validate_count    = (df['date_validate']).sum()
    remarks_validate_count = (df['remarks_validate']).sum()
    amt_validate_count     = (df['amt_validate']).sum()

    #print(company_validate_count, invoice_validate_count, date_validate_count,  remarks_validate_count,amt_validate_count )
    #if company_validate_count > 0 or invoice_validate_count > 0 or date_validate_count > 0 or remarks_validate_count > 0 or amt_validate_count > 0:
    
    return df
    global number_of_success
    number_of_success+=1

    


def df_to_json(df):
        df_list = []
        df=df.reset_index(drop=True)
        for l in range(len(df)):
            exception_list = []
            if df['Company_validate'][l] == True:
                exception_list.append('Company')
            if df['invoice_validate'][l] == True:
                exception_list.append('Invoice Number')
            if df['date_validate'][l] == True:
                exception_list.append('Invoice Date')
            if df['remarks_validate'][l] == True:
                exception_list.append('Remarks')
            if df['amt_validate'][l] == True:
                exception_list.append('Amount paid(AED)')
            dist = {'Company': df['Company'][l], 'Invoice Number': df['Invoice Number'][l], 'Invoice Date': df['Invoice Date'][l], 'Remarks': df['Remarks'][l], 'Amount paid(AED)': df['Amount paid(AED)'][l], 'Exception': exception_list}
            df_list.append(dist)
        return df_list


def find(a, b, conf, thresh):
    text=b
    a=a.split()
    b=b.split()
    c=conf.split()
    c1=c
    flag=0
    for m in range(len(b)):
        if(a==b[m:(m+len(a))]):
            c=c[m:(m+len(a))]
            flag=1
            text=' '.join(b[(m+len(a)):])
            conf=' '.join(c1[(m+len(a)):])
            break
    if(flag==1):
        for n in range(len(c)):
            if(int(c[n])<thresh):
                return 'False',text,conf
    return 'True',text,conf


def Excel(df, writer, n, flow='OUTPUT'):
    if flow == 'OUTPUT':
        df.columns=['Company', 'Invoice Number','Invoice Date','Remarks','Amount paid(AED)','Validate','date_val','amt_val']
    else:
        df.columns=['Company','Invoice Number','Invoice Date','Remarks','Amount paid(AED)']
    df.to_excel(writer, sheet_name=n[-1][2:],header =True, index = False)
    workbook = writer.book
    worksheet = writer.sheets[n[-1][2:]]
    format1 = workbook.add_format({'bg_color': 'red'})
    if flow =='OUTPUT':
        for index, row in df.iterrows():
            if(row['Validate']=='False'):
                worksheet.conditional_format('B'+str(index+2), {'type':'no_blanks','format': format1})
            if(row['date_val']=='False'):
                worksheet.conditional_format('C'+str(index+2), {'type':'no_blanks','format': format1})
            if(row['amt_val']=='False'):
                worksheet.conditional_format('E'+str(index+2), {'type':'no_blanks','format': format1})


def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)




def rmv_punc(word):
    word1=word
    for jj in range(len(word)):
        punctuation='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        if(word[jj] in punctuation):
            word1=word[(jj+1):]
        else:
            break
    return word1


#===================

# Confidence Level Generation for the table
def makeTableConf_pd(text1):
    #text1 = pytesseract.image_to_data(crop_img_table,config='-c preserve_interword_spaces=1 --psm 6')
    lines = text1.split("\n")
    column = lines.pop(0)
    linepd1 = pd.DataFrame([x.split('\t') for x in lines])
    linepd1.columns = ['level','page_num', 'block_num', 'par_num', 'line_num', 'word_num','left','top','width','height','conf','text']
    linepd1 = linepd1[linepd1.text != '']
    linepd1 = linepd1.drop(['level','page_num','block_num','par_num','line_num','word_num','left','top','width','height'], axis=1)
    return linepd1


#==================
def invoiceDateConfCheck(date,conf):
    date = date.strip()
    try:
        datetime_object = datetime.strptime(date, '%d-%m-%y')
        if datetime_object<datetime.now():
            if date[0] in ['0','1','2','3']:
                if date[0]=='3' and not date[1] in [0,1]:
                    return True
                elif int(date[3:4])>12:
                    return True
                else:
                    return(findconfoflist(date,conf))
            else:
                return True
        else:
            return True
    except:
        return True

def  checkAmountConf(amt,amtconfVal):
    if amtconfVal==False:
        #check for amount validity
        if amt[0] in punctuation:
            amtconfVal=True
        elif amt.find('.')==-1:
            amtconfVal=True
        elif amt.count('.')>1:
            amtconfVal=True
        else:
            for i in amt:
                if i in list(map(chr,range(ord('a'),ord('z')+1))) or \
                    i in list(map(chr,range(ord('A'),ord('Z')+1))) or \
                    (i in punctuation and i not in [',','.']):
                    amtconfVal=True
    else:
        return amtconfVal
    return amtconfVal


def sort(img_list, name):
    l = []
    for i in img_list:
        val=i.split('_')[0]
        l.append(int(val))
    l.sort()
    for i in range(len(l)):
        l[i] = str(l[i])+"_"+name
    return l


# funtion to get only text on a white page
def White(image):
    hei, wid, _ = image.shape
    white_bg = 255*np.ones_like(image)
    large = image
    # downsample and use it for processing
    rgb=large
    # apply grayscale
    small = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # morphological gradient
    morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
    # binarize
    _, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # connect horizontally oriented regions
    connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(bw.shape, np.uint8)
    # find contours
    contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        roi = large[y:y + h, x:x + w]
        roi1 = large[y:(y + h-3), x:x + w]
        if (h > 18 and w > 10) and h < 200:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
            if(((h/w)<0.9) | ((h/w)>1.1)):
                if((x>(3*wid/5)) & (w>180)):
                    white_bg[y:(y+h-3), x:x+w] = roi1
                else:
                    white_bg[y:y+h, x:x+w] = roi
    return white_bg


def cleanRawStr(rawstr):
    #print(rawstr)
    rawstr=rawstr.replace('~','-')
    rawstr=rawstr.replace('FU3','FUJ')
    #rawstr =re.sub(r'[^a-zA-Z0-9  -- \/,. ( ) ~ ]', '', rawstr)
    rawstr =re.sub(r'[^a-zA-Z0-9 --\/,.()~]', '', rawstr)
    rawstr=rawstr.strip().lstrip(punctuation).strip()
    listf=[i.strip() for i in rawstr.split()]
    in1=''
    for index,i in enumerate(list(listf)):
        if i[0] in list(map(chr,range(ord('a'),ord('z')+1))) and i[-1] in list(map(chr,range(ord('a'),ord('z')+1))) :
            in1=i
        else:
            break
    if len(in1)>0:
        indexofin1 = rawstr.find(in1)
        rawstr=rawstr[rawstr.find(in1)+len(in1):]
    rawstr=rawstr.strip().lstrip(punctuation).strip()
    return rawstr


def cleanlist(listf):
    listf=[i.strip() for i in listf if i ]
    listf=[i for i in listf if i]
    listf=[i.strip() for i in listf if not i in punctuation]
    #print(listf)
    indexofjunk=[]
    
    for i in listf:
        #print('i',i)
        #print('i.split()',i.split())
        if len(i)<5:
            if all(x.islower() for x in i.split()) :
                #print('YES',i)
                #get the index of it
                indexofjunk.append(listf.index(i))
    #print(indexofjunk)

    if len(indexofjunk)>0:
        for ele in sorted(indexofjunk, reverse = True):
            del listf[ele]

    try: 
        if listf[0].endswith('Total') or listf[0].endswith('total') or  listf[0].endswith('tatal'):
            aa=[listf[0].rpartition(' ')]
            #print(aa)
            out = [item for t in aa for item in t]
            out = [x for x in out if len(x.strip()) > 0]
            listf.pop(0)
            newl=out+listf
            listf=newl
    except:
        print('nil')

    newlist=[]
    for i in listf:
        no_lowercase = ' '.join([word for word in i.split(' ') if not word.islower() or len(word)>5])
        #print(no_lowercase)
        newlist.append(no_lowercase)

    listf = newlist
    '''
    if len(listf[0].split())==1 and listf[0][1].islower():
        listf=listf[1:]
    
    if len(listf[0].split())==1 and listf[0][0].islower():
        listf=listf[1:]
    '''

    if len(listf[0].strip())<=4 and not listf[0].isdigit() :
        listf = listf[1:]
    #print('listf---',listf)
    if len(listf) >= 1:
        if len(listf[0].strip().split()) == 1:
            #print(listf[0])
            if len(listf[0])>1 and len(listf[0]) < 4 :
                if listf[0][1].islower():
                    listf=listf[1:]
                elif listf[0][-1].strip() in punctuation:
                    listf=listf[1:]
                elif listf[0][0].islower():
                    if not listf[0][1].isdigit():
                        listf = listf[1:]
            elif len(listf[0]) == 1:
                listf=listf[1:]
    #print('listf---',listf)
    #Remove junk between Total and amount
    try:
        if listf.index('Total'):
            if not listf[::-1].index('Total') == 1 and listf[::-1].index('Total')==2:
                if listf[::-1][1].isalpha():
                    #print('Y')
                    listf.pop(2)
    except:
        pass
    # printing modified list
    #print (listf)
    if listf and len(listf)>2:
        listf[1]=listf[1].strip().strip(punctuation).strip()
        if '/'in listf[1]:
            listf[0]=listf[0]+listf[1]
            del listf[1]

    return listf

def findconfoflist(str_name,dfconfidence):
    list_conf=[]
    for i in str_name.split():
        list_conf.append(findConf(dfconfidence,i))
    #print(list_conf)
    if False in list_conf or None in list_conf:
        return True
    else:
        return False


#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
def findConf(confdf,text):
    try:
        if text =='':
            return True
        else:
            val=confdf[confdf['text']==text].conf.get_values()[0]
            print(text,val)
            if int(val)>=int(con):
                return True
            else:
                return False
    except:
        False

def match_func(word):
    names=['HM AE AJM Ajman City Ctr',
            'HM AE DUB Deira City Ctr',
            'HM AE ABD Airport Rd(Saqr)',
            'HM AE RAK Manar Mall',
            'HM AE ABD Marina Mall',
            'HM AE SHA Sharjah City Ctr',
            'HM AE DUB Shindagha',
            'HM AE DUB Century Mall',
            'HM AE DUB MOE',
            'HM AE RAK Al Naeem City Ctr',
            'HM AE ABD Dalma Mall',
            'HM AE DUB Mirdif City Ctr',
            'HM AE DUB Madina Mall',
            'HM AE ABD Baniyas',
            'HM AE FUJ Safeer Fujairah',
            'HM AE FUJ Fujairah City Ctr',
            'HM AE ABD Deerfield',
            'HM AE DUB Burjuman',
            'HM AE AJM CITY LIFE AL TALLAH',
            'HM AE DUB Meaisem City Ctr',
            'HM AE DUB Festival City',
            'HM AE DUB Wafi Mall',
            'HM AE AIN Al Ain Mall',
            'HM AE DUB City Land',
            'HM AE DUB Deira Ghurair',
            'UAE DC C4 Platform',
            'HM AE AIN Al Jimmy Mall',
            'HM AE SHA Sharjah City Ctr',
            'HM AE ABD Masdar MAFP',
            'DG AE C4 Platform',
            'HM AE DUB Ibn Batuta',
            'HM AE ABD Yas Island',
            'HM AE DUB Drag Mart',
            'HM AE DUB Bulk Sales',
            'DG AE Amazon Partnership',
            'DG AE Noon Partnership'
             ]
    if(len(difflib.get_close_matches(word, names))>0):
        new=difflib.get_close_matches(word, names)
        score = difflib.SequenceMatcher(None, word, new[0]).ratio()
        if(float(score)>0.90):
            return new[0]
        else:
            print('no')
            return word
    else:
        return word



def main(address,name):   
    print('entereed')
    #try:
    templocation='/media/invoice_temp'
    file1=name
    print(file1)
    name=file1[:-4]
    file = address
    if(file.endswith(".pdf") | file.endswith(".PDF")):
        result = pdf2jpg.convert_pdf2jpg(file,templocation, dpi=300, pages="ALL")
        t=file
    page = 0
    Original_images = []
    n = []
    fi = os.path.join(templocation, file1+'_dir')
    if not os.path.exists(fi):
        fi = os.path.join(templocation, file1+'dir')
    image_list=os.listdir(fi)
    pdf_jpg_name="_".join(image_list[0].split("_")[1:])
    image_list=sort(image_list,pdf_jpg_name)
    print(image_list)
    for file in image_list:
        if (file.endswith(".jpg")) & (page > 0):
            Original_images.append(cv2.imread(os.path.join(fi, file)))
            n.append(file[0:(len(file)-4)])
        page += 1
    # copyDirectory(templocation, path+'/')
    shutil.rmtree(fi)
    img_gr = White(Original_images[-1])
    img_gr = cv2.cvtColor(img_gr, cv2.COLOR_BGR2GRAY)
    img = img_gr
    string = pytesseract.image_to_string(img, lang=lan, config='--psm 6')

    if (('Remarks' not in string) & ('Invoice Date' not in string)&('Invoice Number' not in string)&('Company' not in string)):
        Original_images = Original_images[:-1]
        n = n[:-1]
    if('TAX' in string):
        Original_images = Original_images[:-1]
        n = n[:-1]
    if len(Original_images)>0:
        return main_func(Original_images,  n, file1,fi )
    
    #except:
        
       
    #    print('')