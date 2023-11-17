import pandas as pd
import jieba.posseg as pseg
import jieba
import codecs
from gensim import corpora
from gensim.summarization import bm25
import json
import requests
import fcntl
import pickle


class DocBase():
    def __init__(self,df,bm25,db_id,ftype) -> None:
          self.df=df
          self.db_id=db_id
          self.bm25=bm25
          self.ftype=ftype

    def __search_BM25(self,query,ref_num):
        print('searching bm25...')
        query=jieba.lcut(query)
        scores = self.bm25.get_scores(query)
        #FIXME:concurrent situation may have error
        self.df['similarity']=scores
        print('searching done.')
        return self.df.sort_values("similarity", ascending=False, ignore_index=True).head(ref_num)
    def __search(self,text,ref_num):
        refs=self.__search_BM25(text,ref_num)
        return refs
    def getRef(self,gtype,query,ref_num):
        if gtype=='head':
            refs=self.df.head(ref_num)
        else:
            refs=self.__search(query,ref_num)
        return refs
    def getAllFile(self):
        return self.df.drop_duplicates(subset='file_id', keep='first').rename(columns={'file_id':'id','file_name':'name'}).to_dict(orient='records')
class DocHandler():     
    # store docs and bm25 model
    def handle(self,db_path,stopword_path,ftype)->None:
        data=pd.read_pickle(db_path)
        self.df=self.__paper_df(data)
        self.__build_BM25(stopword_path)
        #文档库建立docbase
        return DocBase(self.df,self.bm25,data['db_id'],ftype)
    def __paper_df(self,data):
        #主要改动将原来的单个文档的dataframe拼接成多个文档的dataframe，并且新增了一列file_name用以指示回答出自哪个文档
        print('creating dataframe')
        file_list=data['file_list'] 
        df_final=pd.DataFrame()
        for item in file_list:
            df=pd.DataFrame(item['file'])
            df['file_name']=item['name']
            df['file_id']=item['id']
            df_final=pd.concat([df_final,df])
        df=df_final
        df.drop_duplicates(subset=['text','page'],keep='first')
        df['length']=df['text'].apply(lambda x:len(x))
        print('Done creating dataframe')
        df['idx'] = range(len(df))
        return df
    def __tokenization(self,text):
        result = []
        words = pseg.cut(text)
        for word, flag in words:
            if flag not in self.stop_flag and word not in self.stopwords:
                result.append(word)
        return result
    def __build_BM25(self,stopword_path):
        print('start build bm25...')
        stopwords = codecs.open(stopword_path,'r',encoding='utf8').readlines()
        self.stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
        self.stopwords = [ w.strip() for w in stopwords ]
        corpus = []
        self.df['wordbag']=self.df.text.apply(lambda x:self.__tokenization(x))
        word_list=self.df.wordbag.tolist()
        for i in word_list:
            corpus.append(i)
        dictionary=corpora.Dictionary(corpus)
        print(len(dictionary))
        self.bm25=bm25.BM25(corpus)
        print('build bm25 done.')

class UserBase():
    def __init__(self,user_id) -> None:
        self.user_id=user_id
        self.msgs=[]
    def clearMsgs(self):
        self.msgs=[]
    def addTurn(self,q,a):
        self.msgs.append({"role":"user","content":q})
        self.msgs.append({"role": "assistant", "content":a})

class ChatGPT():
    def __init__(self,ref_num=3,date='0',url_13b='http://127.0.0.1:50003',url_gpt='http://10.176.64.118:40004') -> None:
        self.ref_num=ref_num
        self.date=date
        self.url_13b=url_13b
        self.url_gpt=url_gpt
        self.blen=1000
    
    def __ask_with_context(self,text,msgs,user:UserBase):
        msgs.append({"role":"user","content":text})
        response=requests.post(f'{self.url_gpt}/ans', json={'messages': msgs,'name_key':'TD8NoNGKkpqBprA89zoOHwTyRTdYeNgNFmRpkPUxglNh6c5zwwK473mIyvMEHW54jz2HonHAuTcrZmCkn/lzFg=='}).json()
        msgs.pop()
        ans=response['ans'][0]
        suc=response['success']
        if not suc:return ans
        print(f">>>>>>>>>>>>\n{text}\n{ans}\n<<<<<<<<<<<<<\n")
        return ans
    def __ask_solo(self,text,user:UserBase):
        response=requests.post(f'{self.url_gpt}/ans', json={'messages': [{"role":"user","content":text}],'name_key':'TD8NoNGKkpqBprA89zoOHwTyRTdYeNgNFmRpkPUxglNh6c5zwwK473mIyvMEHW54jz2HonHAuTcrZmCkn/lzFg=='}).json()
        ans=response['ans'][0]
        suc=response['success']
        if not suc:return ans
        print(f">>>>>>>>>>>>\n{text}\n{ans}\n<<<<<<<<<<<<<\n")
        return ans
    def __ask_solo_7b(self,text,loratype,user:UserBase):
        print("QQQ")
        response = requests.post(f'{self.url_13b}/ans', json={'query': text,'loratype':loratype}).json()
        ans=response['ans']
        suc=response['success']
        if not suc:return ans
        print(f">>>>>>>>>>>>\n{text}\n{ans}\n<<<<<<<<<<<<<\n")
        return ans
    def ans_with_short_passage(self,text,refs,user:UserBase,model_type='chatgpt'):
        ref_list=[k['text'] for k in refs]
        prex=f"参考这一篇文章里与问题相关的以下{len(ref_list)}段文本，然后回答后面的问题：\n"
        for i,ref in enumerate(ref_list):
            prex+=f"[{i+1}]:{ref}\n"
        query=f"{prex}\n你应当尽量用原文回答。若文本中缺乏相关信息，则回答“没有足够信息来回答”。问题：{text}\n："
        #FIXME:
        if model_type=='cutegpt':
            ans=self.__ask_solo_7b(query,'qa',user)
        else:
            ans=self.__ask_solo(query,user)
        return ans.strip(),refs
        # qans=jieba.lcut(ans)
        # scores = file.bm25.get_scores(qans)
        # file.df['similarity']=scores
        # reff = refs.merge(file.df[['idx','similarity']], left_on='idx', right_on='idx', suffixes=('', '_new'))
        # rerank_ref=reff.sort_values("similarity_new", ascending=False, ignore_index=True)
        # refs=rerank_ref[['page','text','file_name','file_id']].to_dict(orient='records')
        # print(ans)
        # return ans.strip(),refs
    def recommend_question(self,user:UserBase,file:DocBase,model_type='chatgpt'):
        #前n段
        refs=file.getRef('head','',self.ref_num)
        text=""
        ref_list=refs.text.tolist()
        ref_page=refs.page.tolist()
        prex=f"参考这一篇文章的前{len(ref_list)}段文本，简要的多方面的概括文章提到了哪些内容，并生成3个推荐问题并用序号列出（推荐问题应该能根据文章的内容回答）：\n"
        for i,ref in enumerate(ref_list):
            prex+=f"{i+1}:{ref}\n"
        if model_type=="cutegpt":
            ans=self.__ask_solo_7b(f"{prex}\n{text}\n：",'head',user)
        else:
            ans=self.__ask_solo(f"{prex}\n{text}\n：",user)
        return f"你好，欢迎和我聊关于文档的任何事！{ans.strip()}"
    def handle_question(self,query,user:UserBase,file:DocBase):
        # query=query.strip()
        # refs=file.getRef('retr',query,self.ref_num)
        # return "",refs
        pass
