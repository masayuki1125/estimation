import multiprocessing
import multiprocessing.pool
import numpy as np

class MC():
    def __init__(self):
        super().__init__()
        
        self.TX_antenna=1
        self.RX_antenna=1
        self.MAX_ERR=8
        self.EbNodB_start=-5
        self.EbNodB_end=1
        self.EbNodB_range=np.arange(self.EbNodB_start,self.EbNodB_end)

    @staticmethod
    def output(inputs):
        '''
        #あるSNRで計算結果を出力する関数を作成
        #main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        '''
        main_func,\
        EbNodB\
        =inputs
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


    def monte_carlo(self,main_func):
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
        
        for i,EbNodB in enumerate(self.EbNodB_range):
            inputs=[(main_func,EbNodB)]
            
            #multiprocess
            #if __name__ == "__main__":
            pool = multiprocessing.Pool(8) # プロセス数を設定
            result=pool.map(self.output, inputs*self.MAX_ERR)  # 並列演算
            #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
            pool.close()
            pool.join()

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
