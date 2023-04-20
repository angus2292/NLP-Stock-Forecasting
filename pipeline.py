import numpy as np
import pandas as pd
import monpa
import os
from monpa import utils
from glob import glob
from dateutil.relativedelta import relativedelta
from datetime import datetime
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class Pipeline:
    def __init__(self, text_df_name, stock_df_name, voc_key, stock_key, start_time, end_time , time_shifts, threshold = 10):
        # 初始化參數
        self.text_df_name = text_df_name
        self.stock_df_name = stock_df_name
        self.voc_key = voc_key
        self.stock_key = stock_key
        self.strat_time = start_time
        self.end_time = end_time
        self.time_shifts = time_shifts
        self.threshold = threshold
        
        self.loaded_data = None
        self.text_file = None
        self.stock_file = None
        self.text_df = None
        self.stock_df = None
        self.months = None
        self.month_index = None
        self.training_df = None
        self.val_df = None
        
        self.X_train = None
        self.X_test = None
        self.y_trian = None
        self.y_test = None
        self.y_pred = None
        self.val_x = None
        self.val_y = None  
              
        self.model = None

    def generating_months(self):
        self.months = pd.period_range(self.strat_time, self.end_time, freq= 'M')
        self.months = self.months.to_timestamp()
        
    def training_info(self):
        print(f'Numbers of Training data: {self.X_train.shape}')
        print(f'Numbers of Testing data: {self.X_test.shape}')
        print(f'Numbers of Forcasting data: {self.val_x.shape}')
    
    def model_performance(self):
        '''
        評估模型績效, 包含CV acc, average acc, report, confusion_matrix
        '''
        print('----Trainig ----')
        print(f"ACC: {round(accuracy_score(self.y_train, self.model.predict(self.X_train)), 4)} | Recall :{round(recall_score(self.y_train, self.model.predict(self.X_train), average = 'macro'), 4)}" )
        print('---- Testing ----')
        print(f"ACC: {round(accuracy_score(self.y_test, self.model.predict(self.X_test)), 4)} | Recall :{round(recall_score(self.y_test, self.model.predict(self.X_test), average = 'macro'), 4)}" )
        print(f'- Confusion_Matrix - \n {pd.DataFrame(np.array(confusion_matrix(self.y_test, self.y_pred, labels=[0,1])))}')
        print('----Forcasting Set----')
        print(f"ACC : {round(accuracy_score(self.val_y, self.val_y_pred), 4)} | Recall: {round(recall_score(self.val_y, self.val_y_pred , average = 'macro'), 4)}")
        # print(f'Report : \n{classification_report(self.val_y, self.val_y_pred)}')
        print(f'- Confusion_Matrix - \n {pd.DataFrame(np.array(confusion_matrix(self.val_y, self.val_y_pred, labels=[0,1])))} \n')
        
    def cut_voc(self, df_col):
        strs = ''
        cut_result = monpa.cut(df_col)
        for item in cut_result:
            item = item.strip()
            # 字長大於1或小於等於4
            if (len(item)>1) & (len(item) <= 4):
                strs = strs + ' '+item
        return strs
        
    def load_data(self):
        temp = []
        for index in range(len(self.text_df_name)):
                temp.append(pd.read_csv(self.text_df_name[index])[['id', 'title', 'content','p_type', 's_name', 'post_time']])
        self.loaded_data = pd.concat(temp, axis=0, ignore_index=True)
                 
    def text_df_processing(self):
        self.text_file = self.loaded_data.copy()
        self.text_file['post_time'] = self.text_file['post_time'].str.slice(0,10)
        self.text_file['post_time'] = pd.to_datetime(self.text_file['post_time'])
        self.text_file.rename(columns={'post_time': '年月日'}, inplace= True)
        self.text_file['words'] = self.text_file['title'] + self.text_file['content']
        self.text_file['words'] = self.text_file['words'].astype(str)
        # 找出包含目標元素的文章
        self.text_file = self.text_file[self.text_file['words'].apply(lambda x: any(list_element in x for list_element in self.voc_key))].reset_index(drop = True)
        
        self.text_file['chunk'] = self.text_file['words'].map(utils.short_sentence)
        self.text_file['chunk'] = self.text_file['chunk'].astype(str)
        
        self.text_df = self.text_file[(self.text_file['年月日'] >= self.months[self.month_index]) &
                                    (self.text_file['年月日'] < self.months[self.month_index] + relativedelta(months= self.time_shifts))].reset_index(drop = True)
        
        print(f'訓練資料日期: {self.months[self.month_index]} ~ {self.months[self.month_index] + relativedelta(months= self.time_shifts)}')
        print(f'預測資料日期: {self.months[self.month_index]+ relativedelta(months= self.time_shifts)} ~ {self.months[self.month_index]+ relativedelta(months= self.time_shifts+1)}')
        
        # 斷詞
        print(f'Training data \n斷詞資料數: {self.text_df.shape[0]}')
        self.text_df['cut'] = None
        self.text_df['cut'] = self.text_df.chunk.apply(lambda x: self.cut_voc(x))
        print('斷詞結束...')

        # 選出可能會用的features
        # self.text_df = self.text_df[['id', 'p_type', 's_name', '年月日', 'cut']]
        self.text_df = self.text_df[['id','年月日', 'cut']]
        return self.text_df
    
    def stock_df_processing(self):
        self.stock_file = pd.read_csv(self.stock_df_name,low_memory=False)
        
        # 將日期轉為datetime
        self.stock_file['年月日'] = pd.to_datetime(self.stock_file['年月日'])
        
        # 找出包含目標的股票
        self.stock_file = self.stock_file[self.stock_file['證券代碼'] == self.stock_key].reset_index(drop = True)
        
        # 轉換資料格式
        self.stock_file['開盤價(元)'] = self.stock_file['開盤價(元)'].astype(float)
        self.stock_file['最高價(元)'] = self.stock_file['最高價(元)'].astype(float)
        self.stock_file['最低價(元)'] = self.stock_file['最低價(元)'].astype(float)
        # 特徵工程
        # 該處是將當天收盤價 - 當天開盤價, 若大於零則視為看漲, 反之則為看跌
        self.stock_file['minus'] = self.stock_file['收盤價(元)'] - self.stock_file['開盤價(元)']
        self.stock_file['漲跌'] = self.stock_file['minus'].map(lambda x: '看漲' if x >=0 else '看跌')
        self.stock_file.drop('minus', axis = 1, inplace = True)
        
        self.stock_df = self.stock_file[(self.stock_file['年月日'] >= self.months[self.month_index]) &
                                       (self.stock_file['年月日'] < self.months[self.month_index] + relativedelta(months= self.time_shifts))].reset_index(drop = True)
        self.stock_df['年月日'] = self.stock_df['年月日'] - pd.Timedelta(days= 1)
        self.stock_df = self.stock_df[['年月日', '漲跌']].sort_values('年月日', ascending=True )
        
        return self.stock_df
       
    def training_data_processing(self):
        # 將處理後stock_df的日期與漲跌mapping回text_df
        # 如 stock_df 3/14看跌則text_df 3/14的文章會被標記為看跌
        self.training_df = pd.merge(left = self.text_df,right =  self.stock_df, on = '年月日', how = 'left')
        y = self.training_df['漲跌'].map(lambda x: 1 if x == '看漲' else 0)
        
        # 建構向量空間
        # 此處的input是斷詞後的每個詞
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(self.training_df['cut'])
        self.featureas = vectorizer.get_feature_names_out()
        
        # 挑選大於 tf-idf threshold的關鍵字
        df_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index = list(self.training_df['id']))
        sums = tfidf.sum(axis = 0)
        terms = vectorizer.get_feature_names_out() 
        data = []
        for col, term in enumerate(terms):
            data.append( (term, sums[0,col] ))
        ranking = pd.DataFrame(data, columns=['term','rank'])
        
        # 此處25為 tf-idf 之 threshold
        ranking =ranking[ranking['rank'] > self.threshold].sort_values('rank', ascending=False).reset_index(drop = True)
        df_tfidf = df_tfidf[ranking['term'].tolist()]
        self.featureas = np.array(df_tfidf.columns)
        
        print(f"訓練看跌文章數: {len(y[y == 0])} | percentage: {len(y[y== 0]) / len(y)}")
        print(f"訓練看漲文章數: {len(y[y == 1])} | percentage: {len(y[y == 1]) / len(y)}")
        
        if abs((len(y[y == 0 ])/ len(y)) - (len(y[y == 1 ])/ len(y))) > 0.3:
            print('具有資料不平衡現象... 進行SMOTE處理...')
            sm = SMOTE()
            df_tfidf, y = sm.fit_resample(df_tfidf, y)
            print(f"After SMOTE 訓練看跌: {len(y[y == 0])} | percentage: {len(y[y== 0]) / len(y)}")
            print(f"After SMOTE 訓練看漲: {len(y[y == 1])} | percentage: {len(y[y == 1]) / len(y)}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_tfidf, y, test_size=0.2)
        else:
        # 切分訓練資料
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_tfidf, y, test_size=0.2)  
                    
    def forcasting_data_procrssing(self):
        val_data = pd.merge(self.text_file, self.stock_file[['年月日', '漲跌']], on='年月日')
        val_data['漲跌'] = val_data['漲跌'].map(lambda x: 1 if x == '看漲' else 0)
        self.val_df = val_data[(val_data['年月日'] >= self.months[self.month_index]+ relativedelta(months= self.time_shifts)) &
                                       (val_data['年月日'] < self.months[self.month_index]+ relativedelta(months= self.time_shifts+1))].reset_index(drop = True)
        print(f'Forcasting data \n斷詞資料數: {self.val_df.shape[0]}')
        self.val_df['cut'] = None
        self.val_df['cut'] = self.val_df.chunk.apply(lambda x: self.cut_voc(x))
        print('斷詞結束...')
        
        print(f"測試看跌文章數: {len(self.val_df[self.val_df['漲跌'] == 0])} | percentage: {round(len(self.val_df[self.val_df['漲跌'] == 0]) / len(self.val_df), 4)}")
        print(f"測試看漲文章數: {len(self.val_df[self.val_df['漲跌'] == 1])} | percentage: {round(len(self.val_df[self.val_df['漲跌'] == 1]) / len(self.val_df), 4)}")
        

        vectorizer = TfidfVectorizer(vocabulary=self.featureas)
        self.val_x = vectorizer.fit_transform(self.val_df['cut'])
        self.val_x = pd.DataFrame(self.val_x.toarray(), columns=self.featureas, index = list(self.val_df['id']))
        self.val_y = self.val_df['漲跌']
                      
    def xgbclassifier(self, GPU = False):
        # 選用XGBoost模型(機器學習的常勝軍)
        # 建模,可以自行選擇要不要用GPU跑, GPU計算比較快
        
        if GPU == True:
            xgb = XGBClassifier(tree_method = 'gpu_hist')
        else:
            xgb = XGBClassifier()
        self.model = xgb.fit(self.X_train, self.y_train)
        # 預測 Testing set
        self.y_pred = xgb.predict(self.X_test)
        # 預測 Forcasting set
        self.val_y_pred = xgb.predict(self.val_x)
        print('XGB:')
        # 績效評估
        self.model_performance()
        # 儲存模型
        joblib.dump(self.model, f'{self.path}/{self.stock_key} XGBoost.pkl')
        
    def svmcalssifier(self):
        svm = SVC()
        self.model = svm.fit(self.X_train, self.y_train)
        self.y_pred = svm.predict(self.X_test)
        self.val_y_pred = svm.predict(self.val_x)
        print('SVM:')
        self.model_performance()
        joblib.dump(self.model, f'{self.path}/{self.stock_key} SVM.pkl')
        
    def rf_classifier(self):
        rf = RandomForestClassifier()
        self.model = rf.fit(self.X_train, self.y_train)
        self.y_pred = rf.predict(self.X_test)
        self.val_y_pred = rf.predict(self.val_x)
        print('RF:')
        self.model_performance()
        joblib.dump(self.model, f'{self.path}/{self.stock_key} Random Forest.pkl')
    
    def dt_classsifier(self):
        dt = DecisionTreeClassifier()
        self.model = dt.fit(self.X_train, self.y_train)
        self.y_pred = dt.predict(self.X_test)
        self.val_y_pred = dt.predict(self.val_x)
        print('DT:')
        self.model_performance()
        joblib.dump(self.model, f'{self.path}/{self.stock_key} CART.pkl')
        
    def knn_classifier(self):
        knn = KNeighborsClassifier()
        self.model = knn.fit(self.X_train, self.y_train)
        self.y_pred = knn.predict(self.X_test)
        self.val_y_pred = knn.predict(self.val_x)
        print('KNN:')
        self.model_performance()
        joblib.dump(self.model, f'{self.path}/{self.stock_key} KNN.pkl')
        
    def proceed(self, GPU):
        self.generating_months()
        self.load_data()
        for index in range(len(self.months)):
            self.month_index = index
            self.text_df_processing()
            self.stock_df_processing()
            self.training_data_processing()
            self.forcasting_data_procrssing()
            self.training_info()
            self.path = f'./Model/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            # training
            self.xgbclassifier(GPU = GPU)
            self.rf_classifier()
            self.dt_classsifier()
            self.knn_classifier()
            self.svmcalssifier()
        
