'''
Created on 2018年1月21日

@author: WZD
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class TS:
    
    def __init__(self,MAX_GEN,length,N,Num,b_0=138,alpha=0.5,beta=0.7,pi=[],ri=[],C_max=0.0,R_max=0.0):
        '''
        MAX_GEN:最大迭代次数
        N:从领域中选去的候选集的个数
        length：禁忌表的长度
        Num:工序的数目,即编码的长度
        b_0:模糊度阈值
        alpha,beta:惩罚系数
        pi:原始顺序的加工时间的主键值
        ri:原始顺序的加工时间的模糊度
        C_max:所有排法中总的加工时间的最大值
        R_max:所有排法中总的加工时间模糊度的最大值
        '''
        self.MAX_GEN = MAX_GEN        
        self.length = length
        self.N = N
        self.Num = Num
        self.b_0 = b_0
        self.alpha = alpha
        self.beta = beta
        self.pi = pi
        self.ri = ri
        self.C_max = C_max
        self.R_max = R_max
        self.P_pi = []
        self.neighbor = [] #邻居
        
        self.Ghh = []                  #当前最佳编码      
        self.current_fitness = 0.0     #当前最佳编码的适应度值
        self.fitness_Ghh_current_list = []  #当前最佳编码的适应度值列表
        self.Ghh_list = []             #当前最佳编码的列表
        
        self.bestGh = []        #最好的编码 
        self.best_fitness = 0.0 #最好编码的适应度值      
        self.best_fitness_list = [] #最好编码的适应度值列表
        self.tabu_list = np.random.randint(0,1,size=(self.length,self.Num)).tolist() #初始化禁忌表
                
    #生成初始解
    def InitialSolution(self):       
        self.Ghh = (np.argsort(self.ri)+1).tolist()  #初始解为将工件按其加工时间模糊度非减次序的排列  
       
    #2-opt邻域交换,得到邻居 
    def swap(self):        
        for i in range(len(self.Ghh)-1):
            for j in range(i+1,len(self.Ghh)):
                temp = self.Ghh.copy()
                temp[i],temp[j] = temp[j],temp[i]
                self.neighbor.append(temp)
        print(self.neighbor)        
    #判断某个编码是否在禁忌表中
    def judgment(self,GN=[]):
        #GN：要判断是否在禁忌表中的交换操作.
        flag = 0 #表示这个编码不在禁忌表中        
        for temp in self.tabu_list:
            temp_reverse = []
            for i in reversed(temp):
                temp_reverse.append(i)
            if GN == temp or GN == temp_reverse:
                flag = 1 #表示这个编码在禁忌表中
                break
        return flag
    
    #更改禁忌表
    def ChangeTabuList(self,GN=[],flag_ = 1):
        #GN：要插入禁忌表的新交换操作，GN=[1,2]
        #flag_:用于判断是否满足藐视原则,flag_ = 1表示满足藐视原则
        if flag_ == 0:
            self.tabu_list.pop()        #弹出最后一个编码
            self.tabu_list.insert(0,GN) #开始位置插入新的编码
        if flag_ == 1:
            for i, temp in enumerate(self.tabu_list):
                temp_reverse = []
                for j in reversed(temp):
                    temp_reverse.append(j)                
                if GN == temp or GN == temp_reverse:
                    self.tabu_list.pop(i)
                    self.tabu_list.insert(0,GN)
        
    #适应度函数（评价函数）
    def fitness(self,GN=[]):
        #GN:要计算适应度函数的编码        
        fitness_pi_ij = 0.0
        #计算加工序GN编码对应的主键值
        p_pi_ij = 0.0
        for i in range(self.Num):
            p_pi_ij = p_pi_ij + (self.Num-i)*self.pi[GN[i]-1] 
        #计算加工序GN编码对应的模糊度
        r_pi_ij = 0.0
        for i in range(self.Num):
            r_pi_ij = r_pi_ij + (self.Num-i)*self.ri[GN[i]-1]       
        #计算适应度    
        if r_pi_ij <= self.b_0:
            fitness_pi_ij = 2*self.C_max + 1 - p_pi_ij     
        elif self.b_0 < r_pi_ij and  r_pi_ij <=((1-self.alpha)*self.R_max+self.alpha*self.b_0):
            fitness_pi_ij = 2*self.C_max + 1 - p_pi_ij -self.beta*self.C_max*(r_pi_ij-self.b_0)/((1-self.alpha)*(self.R_max-self.b_0))         
        elif r_pi_ij >=((1-self.alpha)*self.R_max+self.alpha*self.b_0):
            fitness_pi_ij = self.C_max + 1 - p_pi_ij + (1-self.beta)*self.C_max*(self.R_max - r_pi_ij)/(self.alpha*(self.R_max - self.b_0))
        return fitness_pi_ij   
    
    def solver(self):
        #初始化
        self.InitialSolution()        #生成当前最佳编码self.Ghh
        self.current_fitness = self.fitness(GN = self.Ghh) #self.Ghh的适应度值   
        
        self.bestGh = self.Ghh #复制self.Ghh到最好的编码self.bestGh        
        self.best_fitness = self.current_fitness #最好的适应度值    
        self.best_fitness_list.append(self.best_fitness)
          
        self.Ghh_list.append(self.Ghh.copy()) ##更新当前最佳编码体的列表
        self.fitness_Ghh_current_list.append(self.current_fitness)  #更新当前的最佳适应度值列表         
        
        step = 0 #当前迭代步数      
        while(step<=self.MAX_GEN):            
            self.swap() #产生邻居二维列表self.neighbor,记住后面需要置空            
            #计算每个邻居的适应度函数值
            fitness = []
            for temp in self.neighbor:
                fitness_pi_ij = self.fitness(GN = temp) #传入的一个加工序列的适应度
                fitness.append(fitness_pi_ij)            
            #按照适应度函数值从大到小排定候选次序
            temp = np.argsort(fitness).tolist()
            fitness_sort = [] #适应度排序后的值
            for i in temp:
                fitness_sort.append(fitness[len(fitness)-1-i])                            
            neighbor_sort = [] #按照适应度函数值从大到小排定候选次序后的邻居，第一个邻居的适应度函数值最大
            for i in range(len(temp)):
                neighbor_sort.append(self.neighbor[temp[len(temp)-1-i]])  
            self.neighbor = []  #将邻居二位列表置空，以便下次使用
            
            
            neighbor_sort_N = neighbor_sort[:self.N] #选取邻居中适配值最好的前N个编码
            fitness_sort_N = fitness_sort[:self.N]   #选取邻居中适配值最好的前N个适应度函数值
            
            
            m = 0            
            for temp in neighbor_sort_N:                
                GN = [] #用来装交换的元素GN=[1,2]和GN=[2,1]相同
                for i,temp_Ghh in enumerate(self.Ghh): #self.Ghh:当前最佳编码
                    if temp_Ghh != temp[i]:
                        GN.append(temp_Ghh)        
                                
                flag = self.judgment(GN=GN)            #判断该种互换是否在禁忌表中               
                if flag == 1: #表示这个互换在禁忌表中
                    #判断藐视准则是否满足
                    if fitness_sort_N[m]>self.best_fitness:   #满足藐视规则                        
                        self.current_fitness = fitness_sort_N[m]                     #更新当前最佳适应度函数值
                        self.fitness_Ghh_current_list.append(self.current_fitness)   #更新当前最佳适应度函数值列表                        
                        self.Ghh = neighbor_sort_N[m]     #更新当前最佳编码
                        self.Ghh_list.append(self.Ghh.copy())            #更新当前最佳编码体的列表
                        
                        
                        self.best_fitness = fitness_sort_N[m] #更新最好的适应度函数值
                        self.best_fitness_list.append(self.best_fitness)
                        self.bestGh = temp.copy()             #更新最好的编码
                        #更新禁忌表
                        self.ChangeTabuList(GN=GN, flag_=1)    
                        break
                    else:
                        m = m + 1                          
                else : #表示这个互换不在禁忌表中
                    if fitness_sort_N[0] < self.current_fitness:
                        self.current_fitness = fitness_sort_N[0] #更新当前的最佳适应度值
                        self.Ghh = neighbor_sort_N[0]     #更新当前最佳编码
                        self.Ghh_list.append(self.Ghh.copy())           #更新当前最佳编码体的列表
                        self.fitness_Ghh_current_list.append(self.current_fitness)  #更新当前的最佳适应度函数值列表
                        #更新禁忌表
                        self.ChangeTabuList(GN=GN, flag_=0)
                        break
                    else:
                        self.current_fitness = fitness_sort_N[0] #更新当前的最佳适应度值
                        self.Ghh = neighbor_sort_N[0]      #更新当前最佳编码
                        self.Ghh_list.append(self.Ghh.copy())           #更新当前最佳编码体的列表
                        self.fitness_Ghh_current_list.append(self.current_fitness)   #更新当前的最佳适应度函数值列表
                        #更新禁忌表
                        self.ChangeTabuList(GN=GN, flag_=0)
                        if fitness_sort_N[0]>self.best_fitness:
                            self.best_fitness = fitness_sort_N[0]      #更新最好的适应度函数值
                            self.best_fitness_list.append(self.best_fitness)
                            self.bestGh = neighbor_sort_N[0].copy()    #更新最好的编码
                        break
            P_pi = 0
            for i in range(self.N):
                P_pi = P_pi + (self.N-i)*self.pi[self.Ghh[i]-1] 
            self.P_pi.append(P_pi)    
                             
            step = step + 1
            
            
if __name__ == '__main__':
    df = pd.read_csv('example.txt',sep=' ',index_col=['i'])    
    pi = df.values[:,0].tolist()
    ri = df.values[:,1].tolist()    
    pi_drop = sorted(pi,reverse=True)
    ri_drop = sorted(ri,reverse=True)    
    C_max = 0.0  #所有排法中总加工时间的最大值
    Len = len(pi_drop)
    for i in range(Len):
        C_max = C_max + (Len-i)*pi_drop[i]   
    R_max = 0.0 #所有排法中总加工时间模糊度的最大值
    for i in range(Len):
        R_max = R_max + (Len-i)*ri_drop[i]
    
    ts = TS(MAX_GEN=60,length=5,N=Len,Num=Len,b_0=138,alpha=0.5,beta=0.7,pi=pi,ri=ri,C_max=C_max, R_max=R_max) 
    ts.solver()   
    
    print('最好的适应度函数值：%.2f'%ts.best_fitness)
    print('最好的加工顺序：',ts.bestGh)
    P_pi = 0.0 #最好加工顺序加工时间的总主键值
    R_pi = 0.0 #最好加工顺序加工时间的总模糊度
    for i in range(Len):
        P_pi = P_pi + (Len-i)*pi[ts.bestGh[i]-1]
        R_pi = R_pi + (Len-i)*ri[ts.bestGh[i]-1]
    print('最好加工顺序的总主键值：%.2f'%P_pi)
    print('最好加工顺序的总模糊度值：%.2f'%R_pi)
    
    P= []
    
    for temp in ts.Ghh_list:        
        P_PI = 0.0
        for i in temp:
            P_PI = P_PI + (Len-i)*pi[i-1]
        P.append(P_PI)   
    
    #===========================================================================
    # 
    # plt.plot(ts.fitness_Ghh_current_list,color='blue')    
    # #plt.plot(P,color='yellow')
    # #plt.ylim(min(P),max(P))
    # plt.xlabel(r'迭代步数$i$')
    # plt.ylabel(r'$fitness({\pi _i})$')
    # plt.title('适应度函数值随迭代步数的变化')
    # plt.show()
    #===========================================================================   
     
    plt.plot(ts.P_pi,color='blue')
    plt.xlabel(r'迭代步数$i$')
    plt.ylabel(r'${\rm{P}}\left( {{\pi _i}} \right)$')    
    plt.show()









