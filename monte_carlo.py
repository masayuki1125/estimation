#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import ray
import sys
sys.path.append("../channel")
from AWGN import _AWGN
sys.path.append("../polar_code") #毎回書き換える
from polar_code import coding

cd=coding(1024)
ch=_AWGN()


# In[45]:


'''
#ray.initを一回だけ実行する
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@run_once
def onece():
    ray.init()
'''
ray.init()


# In[47]:


#毎回書き換える関数その2
@ray.remote
def output(K,EbNodB):
        '''
        #あるSNRで計算結果を出力する関数を作成
        #main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        '''

        #import module 
        sys.path.append("../polar_code")
        import polar_code
        pc=polar_code.polar_code(K)
        main_func=pc.polar_code

        '''
        ここから上は毎回書き換え
        '''

        #seed値の設定
        np.random.seed()

        #prepare some constants
        MAX_ERR=1
        count_bitall=0
        count_biterr=0
        count_all=0
        count_err=0
        

        while count_err<MAX_ERR:
        #print("\r"+str(count_err),end="")
            information,EST_information=main_func(EbNodB)
            
            #calculate block error rate
            if np.any(information!=EST_information):
                count_err+=1
            count_all+=1

            #calculate bit error rate 
            count_biterr+=np.sum(information!=EST_information)
            count_bitall+=len(information)

        return count_err,count_all,count_biterr,count_bitall


# In[48]:


class MC():
    def __init__(self):
        super().__init__()
        
        self.TX_antenna=1
        self.RX_antenna=1
        self.MAX_ERR=8
        self.EbNodB_start=-5
        self.EbNodB_end=3
        self.EbNodB_range=np.arange(self.EbNodB_start,self.EbNodB_end) 


# In[49]:


class MC(MC):
    def monte_carlo(self,K):
        '''
        input:main_func
        -----------
        main_func: must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        -----------
        output:BLER,BER

        '''

        BLER=np.zeros(len(self.EbNodB_range))
        BER=np.zeros(len(self.EbNodB_range))

        print("from"+str(self.EbNodB_start)+"to"+str(self.EbNodB_end))
        
        result_ids=[[] for i in range(len(self.EbNodB_range))]

        for i,EbNodB in enumerate(self.EbNodB_range):
            
            for j in range(self.MAX_ERR):
                #multiprocess    
                result_ids[i].append(output.remote(K,EbNodB))  # 並列演算
                #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列

        for i,EbNodB in enumerate(self.EbNodB_range):

            result=ray.get(result_ids[i])

            count_err=0
            count_all=0
            count_biterr=0
            count_bitall=0
            
            for j in range(self.MAX_ERR):
                tmp1,tmp2,tmp3,tmp4=result[j]
                count_err+=tmp1
                count_all+=tmp2
                count_biterr+=tmp3
                count_bitall+=tmp4

            BLER[i]=count_err/count_all
            BER[i]=count_biterr/count_bitall

            if count_biterr/count_bitall<10**-5:
                print("finish")
                break

            print("\r"+"EbNodB="+str(EbNodB)+",BLER="+str(BLER[i])+",BER="+str(BER[i]),end="")
        return BLER,BER

# In[46]:


#毎回書き換える関数その１
class savetxt(coding,_AWGN,MC):
  def __init__(self,N):
    super().__init__(N)

  def savetxt(self,BLER,BER):

    with open(self.filename,'w') as f:

        #print("#N="+str(self.N),file=f)
        print("#TX_antenna="+str(self.TX_antenna),file=f)
        print("#RX_antenna="+str(self.RX_antenna),file=f)
        print("#modulation_symbol="+str(self.M),file=f)
        #print("#MAX_BLERR="+str(self.MAX_ERR),file=f)
        print("#R="+str(self.R),file=f)
        #print("#iteration number="+str(self.L_MAX),file=f)
        print("#EsNodB,BLER,BER",file=f) 
        for i in range(len(self.EbNodB_range)):
            print(str(self.EbNodB_range[i]),str(BLER[i]),str(BER[i]),file=f)


# In[50]:


K=[512,1024,2048]
for K in K:
    print("N=",K)
    mc=MC()
    BLER,BER=mc.monte_carlo(K)
    st=savetxt(K)
    st.savetxt(BLER,BER)


# In[ ]:




