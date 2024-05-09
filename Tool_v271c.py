from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import numpy as np
import copy
import os
import json
import warnings
import multiprocessing

warnings.filterwarnings('ignore')

class Widget:
    '''
    The widget prepared for other classes
    v2.71a,copyright by AntiMage-Janhonho,
    later version will follow by e(2.718281828459045.....),
    Compared to 2.7, we reorganize the function in Widget and other class,
    Compared to 2.71, we add get_VIF_multi to calculate faster,
    most changes are related to the class DataPreprocessing.
    Left to do: mean value encoder
    '''

    def __init__(self):
        pass

    @staticmethod
    def binary_check(data):
        '''
        Check the type of data(y) is 0-1 or not
        :param data: list_like data
        :return: True or False
        '''
        data = pd.Series(data).copy()

        return data.isin([0, 1])

    @staticmethod
    def nan_normalize(x):
        '''
        Normalize the so many type of nan
        :param x: list_like, raw data
        :return: the normalized data
        '''

        x = pd.Series(x).copy()
        x = x.replace([' ', '', 'null', 'Null', 'NULL', 'nan',
                       'np.nan', 'Nan', 'NaN', 'NAN', 'None', 'NONE'], np.nan)

        return x

    @staticmethod
    def get_jump_point(input_array):
        '''
        Calculate the num of jump point & predict the direction of the jump point
        :param input_s: list_like,PD direction
        :return: tuple,the num of jump point & the predicted direction
        '''

        array = pd.Series(input_array).copy()
        negative_diff = array.diff().apply(np.sign)
        negative_diff = negative_diff.iloc[1:-1]
        negative_diff = negative_diff.replace(0, np.nan)
        negative_diff = negative_diff.fillna(method='ffill')

        positive_diff = array.diff(-1).apply(np.sign)
        positive_diff = positive_diff.iloc[1:-1]
        positive_diff = positive_diff.replace(0, np.nan)
        positive_diff = positive_diff.fillna(method='bfill')
        result = negative_diff * positive_diff
        jump_num = len(result[result > 0])

        direction = np.sign(array.diff().apply(np.sign).sum())

        return jump_num, direction

    @staticmethod
    def bin_type_guess(data):
        '''
        Guess the bin_type
        :param data: DataFrame
        :return: type_dict
        '''

        data = pd.DataFrame(data).copy()
        result_dict = {}
        for i in data.columns:
            try:
                data[i].apply(float)
            except Exception:
                result_dict[i] = 'M'
        return result_dict

    @staticmethod
    def get_missing_rate(data):
        '''
        Calculate the missing rate (advise to run nan_normalize before this function)
        :param data:DataFrame, input_data
        :return: Series,the missing rate of each
        '''

        df = data.copy()
        features = list(df.columns)
        missing_rate = pd.Series(np.nan, index=features)
        data_len = data.shape[0]
        for i in features:
            missing_rate[i] = 1 - df[i].count() / data_len

        missing_rate = missing_rate.sort_values()

        return missing_rate

    @staticmethod
    def get_sparse_rate(data, dropna=True):
        '''
        Calculate the sparse rate of the data (advise to run nan_normalize before this function)
        :param data:DataFrame, input_data
        :param dropna: boolean, whether the nan is considered
        :return: Series,the sparse rate of each
        '''

        df = data.copy()
        features = list(df.columns)
        sparse_rate = pd.Series(np.nan, index=features)
        data_len = df.shape[0]
        for i in features:
            part = df[i].copy()
            tmp = part.value_counts(dropna=dropna)
            if len(tmp.index) == 0:
                sparse_rate[i] = -1
            else:
                sparse_rate[i] = tmp.iloc[0] / data_len

        sparse_rate = sparse_rate.sort_values()

        return sparse_rate

    @staticmethod
    def get_VIF(data, features=None):
        '''
        calculate the VIF of the data, if u use the VIF function of statsmodels.api
        ,take care of the intercept
        :param data: Array_like,the raw_data
        :param features: list_like, the features which need to calculate VIF
        :return: VIF,Series, the vif series
        '''

        df = pd.DataFrame(data).copy()
        if features is None:
            features = list(df.columns)
        VIF = pd.Series(np.nan, index=features)
        df['__intercept'] = 1
        for i in features:
            endog = df[i].values
            exog = df.drop(columns=[i]).values
            r_squared = OLS(endog, exog).fit().rsquared
            VIF[i] = 1. / (1. - r_squared)

        return VIF

    @staticmethod
    def get_VIF_multi(data, features=None,jobn=multiprocessing.cpu_count() *0.8):
        '''
        calculate the VIF of the data with multiprocessing, if you use the VIF function of statsmodels.api
        ,take care of the intercept
        :param data: Array_like,the raw_data
        :param features: list_like, the features which need to calculate VIF
        :return: VIF,Series, the vif series
        '''

        df = pd.DataFrame(data).copy()
        if features is None:
            features = list(df.columns)
        VIF = pd.Series(np.nan, index=features)
        df['__intercept'] = 1

        jobn = int(jobn) + 1
        row_s = pd.Series(range(0, len(features)), index=features)
        jobn = min(jobn, len(row_s.index))
        row_cut = pd.qcut(row_s, jobn, labels=range(0, jobn))
        data_list = []
        for i in range(0, jobn):
            data_list.append(list(row_cut[row_cut == i].index))

        mp = multiprocessing.Pool(jobn)
        mplist = []
        for i in range(0, jobn):
            mplist.append(
                mp.apply_async(
                    func=Bins.get_VIF,
                    kwds={'data':df,'features':data_list[i]}))
        mp.close()
        mp.join()

        for result in mplist:
            part_VIF = result.get()
            VIF = VIF.append(part_VIF)

        return VIF

    @staticmethod
    def mean_value_encoder(x, y):
        '''
        supervised method which is used to encode the high
        :param x:
        :param y:
        :return:
        '''
        raise NotImplementedError


class DataPreprocessing(Widget):
    '''
    General preprocess for the data
    v2.71d,copyright by AntiMage-Janhonho,
    later version will follow by e(2.718281828459045.....),
    Compared to 2.7, we finish some DataPreprocessing function.
    Compared to 2.71b, I fix some bugs in drop_feature_vif, and some bugs in drop_feature_corr_approximate
    and some funciton in Widget previously is moved here.
    Compared to 2.71c, I try to accelerate the function drop_feature_vif by multiprocessing.
    Anyway, I won't write any stupid words here, and I realize some truths, keep calm and carry on.
    Talk is cheap, show me your performance.
    '''

    def __init__(self):
        pass

    @staticmethod
    def drop_feature_vif(data, threld=10, rmn=1,multi_flag = True):
        '''
        Use variance_inflation to drop feature of the data
        :param data: DataFrame, input data
        :param threld: float, the threld of the variance inflation
        :param rmn:int, the remove_num each iteration
        :param multi_flag:boolean, whether to use the multiprocessing mode
        :return: list_like,the col left
        '''

        df = data.copy()
        all_cols = pd.Series(df.columns)
        max_iternum = len(all_cols)
        if multi_flag ==False:
            VIF_func = Widget.get_VIF
        else:
            VIF_func = Widget.get_VIF_multi

        for i in range(0, max_iternum):
            vif = VIF_func(df[all_cols])
            vif = vif.sort_values(ascending=False)
            part_vif = vif.iloc[0:rmn]
            drop_vif = part_vif[part_vif > threld]

            if len(drop_vif.index) == 0:
                break

            all_cols = all_cols[-all_cols.isin(drop_vif.index)]

        return all_cols

    @staticmethod
    def drop_feature_corr_accurate(data, threld, mr_drop=0):
        '''
        Use the corr to drop feature of the data accurately, In fact it's the simplified drop_feature_vif
        :param data: DataFrame, input data
        :param mr_drop: int, whether the column is sequenced by the missing rate,0 means not sequenced
        :return: list, the dropped columns
        '''

        df = data.copy()
        if mr_drop == 0:
            missing_rate = DataPreprocessing.get_missing_rate(data)
            df = df[missing_rate.index]

        drop_columns = []
        corr_df = df.corr()
        for x, corr_x in corr_df.iterrows():
            for y, corr in corr_x.iteritems():
                if x == y:
                    break
                if np.fabs(corr) >= threld and (not (y in drop_columns)):
                    drop_columns.append(x)
                    break

        return drop_columns

    @staticmethod
    def drop_feature_corr_approximate(
            data,
            threld,
            finO=200,
            jobn=multiprocessing.cpu_count() *
                 0.8):
        '''
        Use approximate method to drop feature which has high correlation
        :param data: DataFrame, input data
        :param threld: float,the correlation threld which is advised less than 0.8
        :param finO: float or int, the num of features in One Group
        :param jobn: float, the occupied ratio of the cpus
        :return: list,the droped columns
        '''

        df = data.copy()
        jobn = int(jobn) + 1
        drop_col = []
        prob = 1
        max_iter_num = int(0.6 * df.shape[1] / finO) + 1
        for i in range(0, max_iter_num):
            feature = set(df.columns)
            feature = list(feature.difference(drop_col))
            np.random.shuffle(feature)
            df = df[feature]
            gnum = int(df.shape[1] / finO) + 1
            prob = prob * (1 - 1 / gnum)
            row_s = pd.Series(range(0, len(feature)), index=feature)
            gnum = min(gnum, len(row_s.index))
            row_cut = pd.qcut(row_s, gnum, labels=range(0, gnum))
            df_list = []
            for j in range(0, gnum):
                df_list.append(df[row_cut[row_cut == j].index])

            jobn = min(jobn, gnum)
            mp = multiprocessing.Pool(jobn)
            mplist = []
            for j in range(0, gnum):
                mplist.append(
                    mp.apply_async(
                        func=DataPreprocessing.drop_feature_corr_accurate,
                        kwds={
                            'data': df_list[j],
                            'threld': threld,
                            'mr_drop': 0}))
            mp.close()
            mp.join()

            for result in mplist:
                part_drop_col = result.get()
                drop_col = set(drop_col).union(set(part_drop_col))

            if jobn == 1:
                break

            if prob <= 0.3:
                break

        return drop_col

    @staticmethod
    def drop_feature_corr(data, threld, finO=200):
        '''
        The lazy way to drop feature by correlation
        :param data: DataFrame, input data
        :param threld: float, if the the correlation threld which is advised less than 0.8
        :param finO: float or int, the num of features in One Group
        :return: list,the droped columns
        '''

        if data.shape[1] >= finO:
            drop_columns = DataPreprocessing.drop_feature_corr_approximate(
                data, threld, finO)
        else:
            drop_columns = DataPreprocessing.drop_feature_corr_accurate(
                data, threld)

        return drop_columns


class Bins(Widget):
    '''
    Bin Class which can be used to generate bins of the original variable
    v2.7e,copyright by AntiMage-Janhonho,
    later version will follow by e(2.718281828459045.....),
    Compared to v2.0, this version support pandas verison 0.24.x(not 0.25.x!)
    Compared to v2.7, this version add a new param bn_limit
    Compared to V2.7b, this version fix a multi_processing bug in extreme situation
    Compared to V2.7c, this version fix the a prior bug in woe_iv calc, and modify the inf into nan
    Compared to V2.7d, this version add a new coefficient min_bin_prop, and fix a bug in bin replace
    '''

    def __init__(
            self,
            method=3,
            init_num=20,
            end_num=5,
            min_bin_prop=0.03,
            plimit=5,
            nlimit=5,
            ftype=None,
            bn_limit=40):
        '''
        :param method: int,in fact the category,1:Entropy 2:Gini 3:Chi-square 4:Info-value
        :param init_num: int, the init num of bin
        :param end_num: int, the end num of bin
        :param min_bin_prop: float, the proportion of the min bin
        :param plimit: int, the least positive sample num
        :param nlimit: int, the least negative sample num
        :param ftype: str, in fact category('C','M','D'),the type of the data
        :param bn_limit:int, the limit the num of bin
        '''

        self.method = method
        self.init_num = init_num
        self.end_num = end_num
        self.min_bin_prop = min_bin_prop
        self.plimit = plimit
        self.nlimit = nlimit
        self.bn_limit = bn_limit

        self.__bin_stat = pd.DataFrame()
        self.__bin_interval = []
        self.__bin_map = {}
        self.__ftype = ftype

        if method == 1:
            self.split_func = Bins.__Entropy
        elif method == 2:
            self.split_func = Bins.__Gini
        elif method == 3:
            self.split_func = Bins.__Chi_square
        elif method == 4:
            self.split_func = Bins.__Info_value
        else:
            raise NotImplementedError

    @staticmethod
    def __Gini(bin_df, split):
        '''
        Calucate the Gini Gain
        :param bin_df: DataFrame,bin_df
        :param split: string, which columns used to group
        :return: float,the Gini-Gain
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_total = df['total'].sum()
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()

        group = df.groupby(split)
        CG = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_0 / part_sum_total
            p_1 = part_sum_1 / part_sum_total
            m_0 = p_0 * p_0
            m_1 = p_1 * p_1
            CG = CG + part_sum_total * (1 - m_0 - m_1) / df_sum_total

        init_p_0 = df_sum_0 / df_sum_total
        init_p_1 = df_sum_1 / df_sum_total
        init_G = 1 - init_p_0 * init_p_0 - init_p_1 * init_p_1

        delta_G = 1 - CG / init_G

        return delta_G

    @staticmethod
    def __Entropy(bin_df, split):
        '''
        Calucate the Entropy Gain
        :param bin_df: DataFrame bin_df
        :param split: string, which columns used to group
        :return: float,the Entropy-Gain
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_total = df['total'].sum()
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()

        group = df.groupby(split)
        CE = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_0 / part_sum_total
            p_1 = part_sum_1 / part_sum_total
            m_0 = -p_0 * np.log2(p_0)
            m_1 = -p_1 * np.log2(p_1)
            CE = CE + part_sum_total * (m_0 + m_1) / df_sum_total

        init_p_0 = df_sum_0 / df_sum_total
        init_p_1 = df_sum_1 / df_sum_total
        init_E = -((init_p_0) * np.log2(init_p_0) +
                   (init_p_1) * np.log2(init_p_1))
        # the delta_E is a relative value
        Gain_E = 1 - CE / init_E

        return Gain_E

    @staticmethod
    def __Chi_square(bin_df, split):
        '''
        Calucate the Chi-square value, in fact from sklearn.feature_selection import chi2 can solve the problem
        :param bin_df: DataFrame,bin_df
        :param split: string, which columns used to group
        :return: float,the Chi_square value
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()
        df_sum_total = df['total'].sum()
        group = df.groupby(split)
        CS = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_total * df_sum_0 / df_sum_total
            p_1 = part_sum_total * df_sum_1 / df_sum_total
            CS = CS + (part_sum_0 - p_0) ** 2 / p_0 + \
                 (part_sum_1 - p_1) ** 2 / p_1

        return CS

    @staticmethod
    def __Info_value(bin_df, split):
        '''
        Calculate the IV
        :param bin_df: DataFrame bin_df
        :param split: string, which columns used to group
        :return: float,the Info_Value
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()
        group = df.groupby(split)
        IV = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            p_0 = part_sum_0 / df_sum_0
            p_1 = part_sum_1 / df_sum_1
            IV = IV + (p_0 - p_1) * (np.log(p_0 / p_1))

        return IV

    @staticmethod
    def __intervals_to_list(intervals):
        '''
        Tool function used to return the interval edge
        :param intervals: list which the element is the interval
        :return: list,the bin edge
        '''

        intervals = pd.Series(intervals).copy()
        intervals = intervals.replace([' ',
                                       '',
                                       'null',
                                       'Null',
                                       'NULL',
                                       'nan',
                                       'np.nan',
                                       'Nan',
                                       'NaN',
                                       'NAN',
                                       'None',
                                       'NONE'],
                                      np.nan)
        all_list = []
        for i in intervals.dropna():
            all_list.append(i.left)
            all_list.append(i.right)

        all_list.append(np.inf)
        all_list.append(-np.inf)

        bin_interval = sorted(set(all_list))

        return bin_interval

    def __C_binary_split(self, bin_df,total_num):
        '''
        Binary split the whole part
        :param bin_df:
        :return: the split result of one part
        '''

        bin_df = bin_df.copy()
        maxv = np.nan
        result = None
        for i in range(0, len(bin_df.index) - 1):
            tmp = bin_df.copy()
            left = bin_df.ix[i, 'nbins'].left
            split_num = bin_df.index[i].right
            right = bin_df.ix[i, 'nbins'].right
            tmp.ix[0:i + 1, 'nbins'] = pd.Interval(left, split_num)
            tmp.ix[i + 1:, 'nbins'] = pd.Interval(split_num, right)
            value = self.split_func(tmp, 'nbins')
            prop_s = (tmp[0] + tmp[1]).groupby(tmp['nbins']).apply(sum)
            prop = prop_s.min() / total_num
            if (pd.isnull(maxv) or value > maxv) and prop >= self.min_bin_prop:
                maxv = value
                result = tmp

        if pd.isnull(maxv):
            result = bin_df

        return result

    def __C_split(self, bin_df):
        '''
        Split choose one bin to split
        :param bin_df: the bin_df to split
        :return: one-time split result
        '''

        bin_df = bin_df.copy()
        total_num = (bin_df[0]+bin_df[1]).sum()
        group = bin_df.groupby('nbins')
        result = bin_df.copy()
        maxv = np.nan

        tmp_r = []

        for name, part in group:
            tmp = bin_df.copy()
            if len(part.index) > 1:
                part_bin_df = self.__C_binary_split(part,total_num)
                tmp.ix[tmp.index.get_loc(part_bin_df.index[0]):tmp.index.get_loc(
                    part_bin_df.index[-1]) + 1, 'nbins'] = part_bin_df['nbins']
                value = self.split_func(tmp, 'nbins')
                tmp_r.append(value)
                if pd.isnull(maxv) or value > maxv:
                    maxv = value
                    result = tmp

        return result

    def __C_bin_reduce(self, bin_df):
        '''
        Get the bin_dict to reduce the bin
        :param bin_df: dataframe which is raw
        :return the cut which has been combined simply
        '''

        bin_df = bin_df.copy()
        num = 0
        for i in range(1, len(bin_df.index) + 1):
            psum = bin_df.ix[num:i, 0].sum()
            nsum = bin_df.ix[num:i, 1].sum()
            if psum >= self.plimit and nsum >= self.nlimit:
                bin_df.ix[num:i, 'nbin'] = pd.Interval(
                    bin_df.index[num].left, bin_df.index[i - 1].right)
                num = i
            if i == len(bin_df.index) and num < i:
                change_bin = bin_df.ix[num - 1, 'nbin']
                if pd.isnull(change_bin):
                    bin_df.ix[num:i, 'nbin'] = pd.Interval(
                        bin_df.index[num].left, bin_df.index[i - 1].right)
                else:
                    bin_df.ix[num:i, 'nbin'] = pd.Interval(
                        change_bin.left, bin_df.index[i - 1].right)
                    bin_df.ix[bin_df['nbin'] == change_bin, 'nbin'] = pd.Interval(
                        change_bin.left, bin_df.index[i - 1].right)

        bin_dict = bin_df['nbin'].to_dict()

        return bin_dict

    def __C_get_cut_result(self, cut, y):
        '''
        Combine the bin
        :param cut: interval,the cut to combine
        :param y: the label
        :return: the combined cut
        '''

        cut = cut.copy()
        y = pd.Series(y).copy()
        bin_df = pd.crosstab(index=cut, columns=y)
        left = bin_df.sort_index().index[0].left
        right = bin_df.sort_index().index[-1].right
        bin_df['nbins'] = pd.Interval(left, right)

        if self.end_num < len(bin_df.index):
            nbins = 1
            while (nbins < self.end_num):
                bin_df = self.__C_split(bin_df)
                nbins += 1
            bin_df = bin_df.groupby('nbins').sum().sort_index()
        else:
            bin_df = bin_df.drop(columns='nbins')

        return bin_df

    @staticmethod
    def __MD_preprocessing(ftype, x, y, **kwargs):
        '''
        Transform the discrete data into continuous data,
        for D data, we use mean_encoder which include prior prob
        :param x: list_like, raw data
        :param y: list_like, label data
        :return:list_like, preprocessing data
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)
        x.index = pd.Series(x.index).apply(str)
        x = Bins.nan_normalize(x)
        # here nan is not included

        if 'seq' in kwargs.keys():
            tmp = x.value_counts()
            seq = kwargs['seq']
            seq = pd.Series(seq).copy()
            seq = Bins.nan_normalize(seq)
            # because tmp has no nan so drop it in the seq
            seq = seq.dropna()
            tmp = tmp.reindex(seq)
        elif ftype == 'M':
            tmp = x.value_counts()
            tmp = tmp.sort_index()
        elif ftype == 'D':
            tmp = pd.crosstab(x, y)
            tmp['brate'] = tmp[1] / (tmp[0] + tmp[1])
            tmp = tmp.sort_values(by='brate')
        else:
            raise ValueError('Without seq,Only M or D is permitted')

        map_dict = pd.Series(range(0, len(tmp.index)),
                             index=tmp.index).to_dict()
        xnum = x.apply(lambda x: map_dict.get(x, np.nan))

        return xnum, map_dict

    @staticmethod
    def __MD_reprocessing(bin_df, map_dict):
        '''
        Transform the interval into discrete set
        :param bin_df: bin_stat
        :param map_dict: dict like
        :return: the reprocess result
        '''

        bin_df = bin_df.copy()

        tmp = pd.Series(map_dict)
        tmp.name = 'num'
        tmp.index.name = 'OBin'
        tmp = tmp.reset_index()
        mapping = tmp.set_index('num')['OBin'].to_dict()

        result_dict = {}
        for j in range(0, len(bin_df.index)):
            result_dict[bin_df.index[j]] = []

        mkl = sorted(mapping.keys())
        for i in mkl:
            for j in range(0, len(bin_df.index)):
                if i in bin_df.index[j]:
                    result_dict[bin_df.index[j]].append(mapping[i])

        bin_df = bin_df.rename(index=result_dict)

        return bin_df

    def __generate_bin_stat_without_interval_C(self, x, y):
        '''
        Generate the bin_stat of continuous data accroding to the x,y
        :param x:list_like,the continuous x
        :param y:list_like, the label
        :return:DataFrame, bin_stat
        '''
        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        x = x.replace(np.inf, np.nan)
        x = x.replace(-np.inf, np.nan)
        x = x.apply(float)
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)

        try:
            init_cut = pd.qcut(x, self.init_num, duplicates='drop')
            retbin = pd.Series(init_cut.values.categories).sort_values()
            retbin.iloc[0] = pd.Interval(-np.inf, retbin.iloc[0].right)
            retbin.iloc[-1] = pd.Interval(retbin.iloc[-1].left, np.inf)
            init_cut = pd.cut(x, pd.IntervalIndex(retbin))
            init_cut = init_cut.astype(object)
        except IndexError:
            init_cut = x.copy()
            init_cut[pd.notnull(init_cut)] = pd.Interval(-np.inf, np.inf)
            retbin = pd.Series(pd.Interval(-np.inf, np.inf))

        bin_df = pd.crosstab(index=init_cut, columns=y)
        bin_df = bin_df.reindex(retbin)
        bin_df = bin_df.sort_index()
        bin_df = bin_df.fillna(0.0)

        bin_df['nbin'] = np.nan

        bin_dict = self.__C_bin_reduce(bin_df)
        combine_cut = init_cut.map(bin_dict)
        bin_stat = self.__C_get_cut_result(combine_cut, y)

        return bin_stat

    def __generate_bin_stat_without_interval_MD(self, x, y, **kwargs):
        '''
        Generate the bin_stat of MD data accroding to the x,y
        :param x:list_like,the continuous x
        :param y:list_like, the label
        :return:DataFrame, bin_stat
        '''

        x, map_dict = Bins.__MD_preprocessing(self.__ftype, x, y, **kwargs)
        bin_stat = self.__generate_bin_stat_without_interval_C(x, y)
        bin_stat = Bins.__MD_reprocessing(bin_stat, map_dict)

        return bin_stat

    def __generate_bin_stat_without_interval(self, x, y, **kwargs):
        '''
        Generate bin_stat with interval
        :param x: list_like ,the data
        :param y: list_like,the label
        :param interval: list_like,the interval
        :return: DataFrame, bin_stat
        '''

        if self.__ftype == 'C':
            bin_stat = self.__generate_bin_stat_without_interval_C(x, y)
        elif self.__ftype == 'M' or self.__ftype == 'D':
            bin_stat = self.__generate_bin_stat_without_interval_MD(
                x, y, **kwargs)
        else:
            raise NotImplementedError

        return bin_stat

    def __generate_bin_stat_with_interval_C(self, x, y, interval):
        '''
        Generate the bin_stat with continuous data which has interval
        :param x:list_like, raw data
        :param y:list_like, the label
        :param interval: list_like the interval
        :return:DataFrame, bin_stat
        '''

        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        x = x.apply(float)
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)
        interval = pd.Series(interval).copy()
        interval = Bins.nan_normalize(interval)
        interval = interval.dropna()
        interval = list(set(interval).union([-np.inf, np.inf]))
        interval = pd.Series(interval)
        interval = interval.sort_values()
        interval.index = range(0, len(interval.index))
        interval.index = pd.Series(interval.index).apply(str)

        interval_list = []
        for i in range(0, len(interval.index) - 1):
            interval_list.append(pd.Interval(
                interval.ix[i], interval.ix[i + 1]))

        init_cut = pd.cut(x, pd.IntervalIndex(interval_list))
        retbin = pd.Series(init_cut.values.categories).sort_values()

        init_cut = pd.cut(x, pd.IntervalIndex(retbin))
        init_cut = init_cut.astype(object)

        bin_df = pd.crosstab(index=init_cut, columns=y)
        bin_df = bin_df.reindex(retbin)
        bin_df = bin_df.sort_index()
        bin_df = bin_df.fillna(0.0)

        bin_stat = bin_df

        return bin_stat

    def __generate_bin_stat_with_interval_MD(self, x, y, interval, **kwargs):
        '''
        Generate the bin_stat with Distinct(M,D) data which has interval
        :param x:list_like, raw data
        :param y:list_like, the label
        :param interval: list_like the interval
        :return:DataFrame, bin_stat
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)
        x.index = pd.Series(x.index).apply(str)
        x = Bins.nan_normalize(x)
        x = x.fillna('NaN')
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)
        interval = pd.Series(interval).copy()

        if 'seq' in kwargs.keys():
            seq = kwargs['seq']
            seq = Bins.nan_normalize(seq)
            seq = seq.dropna()
            seq_df = pd.DataFrame({'seq': seq, 'num': range(0, len(seq))})
            num_len = int(np.log10(len(seq))) + 2
            seq_df['num'] = seq_df['num'].apply(
                lambda x: '{:0>{}}'.format(x, num_len))
            seq_dict = seq_df.set_index('seq')['num'].to_dict()
            re_seq_dict = seq_df.set_index('num')['seq'].to_dict()
        else:
            seq_dict = None
            re_seq_dict = None

        if seq_dict is not None:
            x_c = x.apply(lambda x: seq_dict.get(x, 'NaN'))
            interval_c = []
            for i in range(0, len(interval)):
                tmp = interval[i]
                if isinstance(tmp, tuple) or isinstance(tmp, list):
                    tmp = pd.Series(tmp).copy()
                    tmp = tmp.apply(str)
                    tmp = Bins.nan_normalize(tmp)
                    tmp = tmp.apply(lambda x: seq_dict.get(x, np.nan))
                    interval_c.append(list(tmp))
                else:
                    tmp_s = pd.Series([tmp]).apply(str)
                    tmp_s = Bins.nan_normalize(tmp_s)
                    tmp_s = tmp_s.fillna('NaN')
                    tmp = tmp_s.iloc[0]
                    tmp = str(tmp)
                    interval_c.append(seq_dict.get(tmp, np.nan))

            interval_c = pd.Series(interval_c)

        else:
            x_c = x.copy()
            interval_c = interval.copy()

        interval_map = {}
        all_key = []

        for i in range(0, len(interval_c)):
            tmp = interval_c[i]
            if isinstance(tmp, tuple) or isinstance(tmp, list):
                tmp = pd.Series(tmp).copy()
                tmp = tmp.apply(str)
                tmp = Bins.nan_normalize(tmp)
                tmp = tmp.sort_values()
                tmp = tmp.fillna('NaN')
                name = []
                for j in range(0, len(tmp)):
                    name.append(tmp.iloc[j])
                    all_key.append(tmp.iloc[j])
                for j in range(0, len(tmp)):
                    interval_map[tmp.iloc[j]] = name
            else:
                tmp_s = pd.Series([tmp]).apply(str)
                tmp_s = Bins.nan_normalize(tmp_s)
                tmp_s = tmp_s.fillna('NaN')
                tmp = tmp_s.iloc[0]
                tmp = str(tmp)
                interval_map[tmp] = [tmp]
                all_key.append(tmp)

        all_x = x_c.unique()

        for i in all_x:
            if i not in all_key:
                interval_map[i] = [i]

        df = pd.DataFrame({'OBin': pd.Series(interval_map)})
        df['strOBin'] = df['OBin'].apply(str)
        str_dict = df['strOBin'].to_dict()

        if len(set(str_dict.values())) >= self.bn_limit:
            raise Exception('Too many bins!')

        x_c = x_c.apply(lambda x: str_dict[x])
        bin_stat = pd.crosstab(x_c, y)

        tmp = df[['OBin', 'strOBin']].copy()
        tmp = tmp.drop_duplicates(subset=['strOBin'])
        tmp = tmp.set_index('strOBin')
        re_dict = tmp['OBin'].to_dict()
        if seq_dict is None:
            bin_stat = bin_stat.rename(index=re_dict)
        else:
            re_dict2 = {}
            for i in re_dict.keys():
                tmp = list(
                    pd.Series(
                        re_dict[i]).apply(
                        lambda x: re_seq_dict.get(
                            x, 'NaN')))
                re_dict2[i] = tmp
            bin_stat = bin_stat.rename(index=re_dict2)

        return bin_stat

    def __generate_bin_stat_with_interval(self, x, y, interval, **kwargs):
        '''
        Generate bin_stat with interval
        :param x: list_like ,the data
        :param y: list_like,the label
        :param interval: list_like,the interval
        :return: DataFrame, bin_stat
        '''

        if self.__ftype == 'C':
            bin_stat = self.__generate_bin_stat_with_interval_C(x, y, interval)
        elif self.__ftype == 'M' or self.__ftype == 'D':
            bin_stat = self.__generate_bin_stat_with_interval_MD(
                x, y, interval, **kwargs)
        else:
            raise NotImplementedError

        return bin_stat

    def __generate_bin_stat(self, x, y, interval, **kwargs):
        '''
        Get the bin_stat of the continuous variable
        :param x: list_like,the feature
        :param y: list_like,the label
        :param input_interval: list_like, user-defined interval
        '''

        x = pd.Series(x).copy()
        y = pd.Series(y).copy()
        y = y.reindex(x.index)

        x = Bins.nan_normalize(x)
        x.index = range(0, len(x.index))
        x.index = pd.Series(x.index).apply(str)

        y = Bins.nan_normalize(y)
        y.index = range(0, len(y.index))
        y.index = pd.Series(y.index).apply(str)

        y = y.reindex(x.index)

        if pd.isnull(y).any():
            raise ValueError("y has NaN")

        if interval is None:
            bin_stat = self.__generate_bin_stat_without_interval(
                x, y, **kwargs)
        else:
            bin_stat = self.__generate_bin_stat_with_interval(
                x, y, interval, **kwargs)
        
        bin_stat.index.name = 'Interval'
        bin_stat = bin_stat.reset_index()

        all_key = []
        for i in range(0, len(bin_stat.index)):
            if isinstance(bin_stat['Interval'].iloc[i], list):
                all_key = all_key + bin_stat['Interval'].iloc[i]
            else:
                all_key.append(bin_stat['Interval'].iloc[i])

        # deal with NaN if NaN is not in the interval
        if 'NaN' not in all_key:
            y1_nan = y.ix[pd.isnull(x)].sum()
            y0_nan = y.ix[pd.isnull(x)].shape[0] - y1_nan

            if self.__ftype == 'C':
                bin_stat = bin_stat.append(pd.DataFrame(
                    {'Interval': ['NaN'], 0: [y0_nan], 1: [y1_nan]}, index=[bin_stat.shape[0]]))
            else:
                bin_stat = bin_stat.append(pd.DataFrame(
                    {'Interval': [['NaN']], 0: [y0_nan], 1: [y1_nan]}, index=[bin_stat.shape[0]]))

        if 0 not in bin_stat.columns:
            bin_stat[0] = np.nan
        if 1 not in bin_stat.columns:
            bin_stat[1] = np.nan

        # in very seldom condition, only one bin is returned,so take care of it
        bin_stat['type'] = self.__ftype

        bin_stat['Bin'] = range(1, bin_stat.shape[0] + 1)
        num_len = int(np.log10(len(bin_stat.index))) + 2
        bin_stat['Bin'] = bin_stat['Bin'].apply(
            lambda x: 'B{:0>{}}'.format(x, num_len))

        if self.__ftype == 'C':
            bin_stat['lower_json'] = np.nan
            bin_stat['upper_json'] = np.nan
            for j in range(0, len(bin_stat.index)):
                if bin_stat.ix[j, 'Interval'] == 'NaN':
                    bin_stat.ix[j, 'lower_json'] = 'NaN'
                    bin_stat.ix[j, 'upper_json'] = 'NaN'
                else:
                    bin_stat.ix[j, 'lower_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'].left)
                    bin_stat.ix[j, 'upper_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'].right)
        else:
            bin_stat['lower_json'] = np.nan
            bin_stat['upper_json'] = np.nan
            for j in range(0, len(bin_stat.index)):
                if bin_stat.ix[j, 'Interval'] == 'NaN':
                    bin_stat.ix[j, 'lower_json'] = 'NaN'
                    bin_stat.ix[j, 'upper_json'] = 'NaN'
                else:
                    bin_stat.ix[j, 'lower_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'])
                    bin_stat.ix[j, 'upper_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'])

        self.__bin_stat = bin_stat.copy()

    @staticmethod
    def __stat_to_interval(bin_stat, ftype):
        '''
        Tool func generate bin_interval from bin_stat
        :param bin_stat: DataFrame, bin_stat
        :param ftype: str,data type
        :return: list,bin_interval
        '''

        bin_stat = bin_stat.copy()
        if ftype == 'C':
            bin_interval = Bins.__intervals_to_list(bin_stat['Interval'])
        elif ftype == 'M' or ftype == 'D':
            bin_interval = list(bin_stat['Interval'])
        else:
            raise NotImplementedError

        return bin_interval

    def __generate_bin_interval(self):

        bin_stat = self.__bin_stat.copy()
        bin_interval = Bins.__stat_to_interval(bin_stat, self.__ftype)
        self.__bin_interval = bin_interval

    @staticmethod
    def __stat_to_map(bin_stat, ftype):
        '''
        Tool func generate bin_map from bin_stat
        :param bin_stat: DataFrame, bin_stat
        :param ftype: str,data type
        :return: dict,bin_map
        '''

        bin_stat = bin_stat.copy()
        if ftype == 'C':
            tmp = bin_stat.set_index('Interval')
            bin_map = tmp['Bin'].to_dict()
        elif ftype == 'M' or ftype == 'D':
            BI = bin_stat[['Interval', 'Bin']]
            bin_map = {}
            for i in range(0, len(BI.index)):
                tmp = BI['Interval'].iloc[i]
                if isinstance(tmp, tuple) or isinstance(tmp, list):
                    for j in tmp:
                        bin_map[j] = BI['Bin'].iloc[i]
                else:
                    bin_map[tmp] = BI['Bin'].iloc[i]
        else:
            raise NotImplementedError

        return bin_map

    def __generate_bin_map(self):

        bin_stat = self.__bin_stat.copy()
        bin_map = Bins.__stat_to_map(bin_stat, self.__ftype)
        self.__bin_map = bin_map

    def generate_bin_smi(self, x, y, interval=None, ftype=None, **kwargs):
        '''
        Get the bin_interval&bin_map which can be used
        :param x: list_like,the feature
        :param y: list_like,the label
        :param ftype: str, C means continuous data,others means discrete data
        :param interval: list_like, user-defined interval
        :param **kwargs: other parameter
        '''

        if ftype is None and self.__ftype is None:
            raise ValueError('Ftype is needed')

        if ftype is not None:
            self.__ftype = ftype

        if Bins.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        self.__generate_bin_stat(x, y, interval, **kwargs)
        self.__generate_bin_map()
        self.__generate_bin_interval()

    def get_bin_info(self):
        '''
        Return the bin_stat,bin_map& bin_interval to the user
        :return: tuple,bin_stat, bin_interval, bin_map
        '''

        bin_stat = self.__bin_stat.copy()
        bin_stat[[0, 1]] = bin_stat[[0, 1]].fillna(0.0)
        bin_stat['total'] = bin_stat[1] + bin_stat[0]
        bin_stat['PD'] = bin_stat[1] / bin_stat['total']
        bin_stat['1_prop'] = bin_stat[1] / bin_stat[1].sum()
        bin_stat['0_prop'] = bin_stat[0] / bin_stat[0].sum()
        bin_stat['total_prop'] = bin_stat['total'] / bin_stat['total'].sum()
        jump_num, direction = Bins.get_jump_point(bin_stat['PD'].iloc[:-1])
        bin_stat['jn'] = jump_num
        bin_stat['direction'] = direction

        bin_stat.index = pd.Series(bin_stat.index).apply(str)
        bin_interval = copy.deepcopy(self.__bin_interval)
        bin_map = copy.deepcopy(self.__bin_map)

        return bin_stat, bin_interval, bin_map

    def value_to_bin(self, x):
        '''
        From init value to bin_num
        :param x: Series,init value
        :return Series,the replaced value
        '''

        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        if self.__ftype == 'C':
            x = x.apply(float)
            tmp = self.__bin_interval
            interval_list = []
            for i in range(0, len(tmp) - 1):
                interval_list.append(pd.Interval(tmp[i], tmp[i + 1]))
            interval_x = pd.cut(
                x, pd.IntervalIndex(interval_list)).astype(
                object)
            interval_x = interval_x.fillna('NaN')
            result = interval_x.apply(lambda x: self.__bin_map[x])
        elif self.__ftype == 'M' or self.__ftype == 'D':
            x = x.apply(str)
            x = Bins.nan_normalize(x)
            x = x.fillna('NaN')
            num_len = len(list(self.__bin_map.values())[0]) - 1
            result = x.apply(
                lambda x: self.__bin_map.get(
                    x, 'B{:0>{}}'.format(
                        0, num_len)))
        else:
            raise NotImplementedError

        return result

    @staticmethod
    def bin_replace(x, interval=None, ftype='C', y=None):
        '''
        Replace the orignal data with the bin
        :param x: list_like, original data
        :param interval: list_like
        :param ftype: str,C or D
        :param y:list_like,the corresponding y
        :return: tuple,result, bin_stat, bin_map
        '''

        x = pd.Series(x).copy()
        if y is None:
            y = pd.Series(0, index=x.index)
        else:
            y = pd.Series(y).copy()

        if Bins.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        tmp = Bins()
        if ftype == 'C':
            if interval is None:
                raise ValueError('No Interval Error!')
            tmp.generate_bin_smi(x, y, interval, ftype='C')
        elif ftype == 'M':
            if interval is None:
                interval = [np.nan]
            tmp.generate_bin_smi(x, y, interval, ftype='M')
        elif ftype == 'D':
            if interval is None:
                interval = [np.nan]
            tmp.generate_bin_smi(x, y, interval, ftype='D')
        else:
            raise ValueError('Ftype Error!')

        bin_stat, bin_interval, bin_map = tmp.get_bin_info()
        result = tmp.value_to_bin(x)

        return result, bin_stat, bin_map

    @staticmethod
    def __woe_iv(x, y, all_bin, lbd=0.5, Nan_bin_C={}):
        '''
        Caculate the woe we use the MAP(Dirichlet priori) not MLE
        :param x: list_like,the feature
        :param y: list_like,the label
        :param all_bin: list_like, all bin
        :param lbd: float,the number which is Dirichlet priori
        :param Nan_bin_C: dict like {'B01':'min','B02':'max','B03':3.14}
        :return: tuple,iv_contribution, woe, xytable
        '''
        x = pd.Series(x).copy()
        y = pd.Series(y).copy()

        if len(y) > 0:
            yvc = y.value_counts()
            yvc = yvc.reindex([0, 1]).fillna(0.0)

            total_0 = yvc[0]
            total_1 = yvc[1]
            xytable = pd.crosstab(x, y)
            xytable = xytable.reindex(columns=[0, 1]).fillna(0.0)

            xytable.index = pd.Series(xytable.index).apply(str)
            if all_bin is not None:
                xytable = xytable.reindex(xytable.index.union(all_bin))
                xytable = xytable.fillna(0.0)

            # MAP instead of MLE
            tab_len = len(xytable.index)
            relative_0 = (xytable[0] + lbd) / (total_0 + lbd * tab_len)
            relative_1 = (xytable[1] + lbd) / (total_1 + lbd * tab_len)
            #relative_0 = (xytable[0]) / (total_0)
            #relative_1 = (xytable[1]) / (total_1)            
            woe = (relative_1 / relative_0).apply(np.log)
            woe.ix[Nan_bin_C.keys()] = np.nan
            for i in Nan_bin_C.keys():
                if Nan_bin_C[i] == 'min':
                    woe.ix[i] = woe.min()
                elif Nan_bin_C[i] == 'max':
                    woe.ix[i] = woe.max()
                elif Nan_bin_C[i] == 'mid':
                    woe.ix[i] = woe.median()
                else:
                    woe.ix[i] = Nan_bin_C[i]
                break

            iv_contribution = (relative_1 - relative_0) * woe
        else:
            iv_contribution = pd.Series()
            woe = pd.Series()
            xytable = pd.DataFrame()

        return iv_contribution, woe, xytable

    @staticmethod
    def woe_iv(
            x,
            y=None,
            all_bin=None,
            report=None,
            split_bin=[],
            lbd=0.5,
            Nan_bin_C={}):
        '''
        Caculate the woe
        :param x: list_like,the feature
        :param y: list_like,the label
        :param all_bin: list_like, all bin
        :param report: dataframe, the woe_iv report
        :param split_bin: list_like,often the NaN bin
        :param lbd: float,the number which is Dirichlet priori
        :param Nan_bin_C: dict like {'B01':'min'} or{'B01':'max'} or{'B01':mid} or {'B01':3.14}
        :return: tuple,result, report, error_flag
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)

        all_bin = pd.Series(all_bin).copy()

        r_split_bin = list(pd.Series(split_bin))

        if report is None:
            if y is None:
                raise ValueError('No report Nor y!')
            else:
                if Bins.binary_check(y) is False:
                    raise ValueError('Value Error of y!')

                iv_series, woe, xytable = Bins.__woe_iv(
                    x, y, all_bin, lbd=lbd, Nan_bin_C=Nan_bin_C)
                result = iv_series.replace([-np.inf, np.inf, np.nan], 0.0)
                iv_values = pd.Series(result.sum(), index=result.index)

                # here sometimes duplicates index may become a big problem!
                split_x = x[-x.isin(r_split_bin)].copy()
                split_y = y.reindex(split_x.index)
                split_iv_series, split_woe, split_xytable = Bins.__woe_iv(
                    split_x, split_y, all_bin[-all_bin.isin(r_split_bin)], lbd=lbd)
                split_result = split_iv_series.replace(
                    [-np.inf, np.inf, np.nan], 0.0)
                split_iv_values = pd.Series(
                    split_result.sum(), index=result.index)

                if len(r_split_bin) == 0:
                    binary_iv_values = np.nan
                else:
                    binary_x = x.replace(list(r_split_bin), 'binary_1').copy()
                    binary_x[binary_x != 'binary_1'] = 'binary_0'
                    binary_iv_series, binary_woe, binary_xytable = Bins.__woe_iv(
                        binary_x, y, ['binary_0', 'binary_1'], lbd=lbd)
                    binary_result = binary_iv_series.replace(
                        [-np.inf, np.inf, np.nan], 0.0)
                    binary_iv_values = pd.Series(
                        binary_result.sum(), index=result.index)

                report = pd.DataFrame({'woe': woe,
                                       'ivc': iv_series,
                                       'iv': iv_values,
                                       'part_iv': split_iv_values,
                                       'binary_iv': binary_iv_values,
                                       '0': xytable[0],
                                       '1': xytable[1],
                                       'total': xytable[0] + xytable[1]})
                report.index.name = 'Bin'
                report = report.reset_index()
                woe_dict = woe.to_dict()
                result = x.apply(lambda x: woe_dict[x])
                error_flag = False
        else:
            report = report.copy()
            report = report.set_index('Bin')
            woe = report['woe'].copy()
            woe_dict = woe.to_dict()
            result = x.apply(lambda x: woe_dict.get(x, np.nan))
            error_flag = pd.isnull(result).any()

        return result, report, error_flag

    @staticmethod
    def generate_raw(
            data,
            y,
            type_dict={},
            bin_initn_dict={},
            bin_endn_dict={},
            min_bin_prop_dict = {},
            lbd=0.5,
            MDP=False,
            Nan_calculate='all',
            Nan_bin_C_dict={}):
        '''
        Generate the report quickly
        :param data: DataFrame,the original data
        :param y: list_like,the label
        :param type_dict: dict_like,each type of the feature
        :param bin_initn_dict: dict_like, the init_bin_num of each feature
        :param bin_endn_dict: dict_like, the end_bin_num of each feature
        :param min_bin_prop_dict: dict_like, the min_bin_prop of each feature
        :param lbd: float,the number which is Dirichlet priori
        :param MDP: bool, whether to process the MD data
        :param Nan_calculate: list, the feature which NaN must be caculated
        ,'all' means the Nan of all the feature must be calculated by Nan itself
        ,[] means None of the feature should be calculated by Nan itself
        ,['F1'] means only F1 should be calculated by Nan itself
        :param Nan_bin_C_dict: dict, give the exact the Nan_bin_C,Nan_calculate is prior to Nan_bin_C_dict
        :return: tuple, all_report, all_change_report, woe_df, bin_df, error_dict
        '''

        data = data.copy()

        if Nan_calculate == 'all':
            Nan_calculate = list(data.columns)

        all_report = pd.DataFrame()
        all_change_report = pd.DataFrame()

        bin_df = data.copy()
        woe_df = data.copy()

        type_dict_predict = type_dict
        specified_col = list(type_dict.keys())
        left_type_dict_predict = Bins.bin_type_guess(
            data.drop(columns=specified_col, errors='ignore'))
        type_dict_predict.update(left_type_dict_predict)

        type_s = pd.Series(type_dict_predict).copy()
        type_s = type_s.reindex(data.columns)
        type_s = type_s.fillna('C')

        error_dict = {}

        for i in range(0, len(data.columns)):
            tmp = Bins(
                init_num=bin_initn_dict.get(
                    data.columns[i], 20), end_num=bin_endn_dict.get(
                    data.columns[i], 5),min_bin_prop= min_bin_prop_dict.get(data.columns[i], 0.03))
            try:
                part = data[data.columns[i]].copy()
                if type_s.iloc[i] != 'C' and MDP == False:
                    interval = [np.nan]
                    tmp.generate_bin_smi(
                        part, y, interval=interval, ftype=type_s.iloc[i])
                else:
                    tmp.generate_bin_smi(part, y, ftype=type_s.iloc[i])

                bin_stat, bin_interval, bin_map = tmp.get_bin_info()
                bin_stat = bin_stat.drop(columns=[0, 1, 'total'])

                result = tmp.value_to_bin(part)

                Nan_bin_C = {}
                split_bin = []
                for m in range(0, len(bin_stat.index)):
                    interval_tmp = bin_stat['Interval'].iloc[m]
                    if isinstance(interval_tmp, list):
                        if 'NaN' in interval_tmp:
                            if len(interval_tmp) == 1:
                                Nan_bin_C = {
                                    bin_stat['Bin'].iloc[m]: Nan_bin_C_dict.get(
                                        data.columns[i], 'max')}
                            split_bin.append(bin_stat['Bin'].iloc[m])
                            break
                    else:
                        split_bin = split_bin + \
                                    list(
                                        bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])
                        Nan_bin_C[list(bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])[
                            0]] = Nan_bin_C_dict.get(data.columns[i], 'max')
                        break

                split_bin = list(set(split_bin))

                if data.columns[i] in Nan_calculate:
                    Nan_bin_C = {}

                woe_result, woeiv, error_flag = Bins.woe_iv(result, y, all_bin=list(
                    bin_stat['Bin']), split_bin=split_bin, lbd=lbd, Nan_bin_C=Nan_bin_C)

                report = pd.merge(
                    bin_stat, woeiv, left_on='Bin', right_on='Bin')
                report['Feature'] = data.columns[i]

                woe_df[data.columns[i]] = woe_result
                bin_df[data.columns[i]] = result
                all_report = all_report.append(report)

                if bin_stat['type'].iloc[0] == 'C':
                    bin_interval = bin_interval[1:-1]

                change_report = pd.DataFrame({'Feature': [data.columns[i]], 'Interval': [
                    bin_interval], 'type': [type_s.iloc[i]]})
                change_report['Interval'] = change_report['Interval'].apply(
                    lambda x: json.dumps(x))
                all_change_report = all_change_report.append(
                    change_report, ignore_index=True)

            except Exception as err:
                error_dict[data.columns[i]] = err

        try:
            all_report = all_report.sort_values(
                by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])

            all_change_report = all_change_report.set_index('Feature')
            all_change_report = all_change_report.reindex(
                all_report.Feature.drop_duplicates())
            all_change_report = all_change_report.reset_index()

            col_list = list(all_report.columns)
            col_list.remove('Feature')
            all_report = all_report[['Feature'] + col_list]

            bin_df = bin_df.drop(columns=list(error_dict.keys()))
            woe_df = woe_df.drop(columns=list(error_dict.keys()))

        except Exception:
            all_report = pd.DataFrame()
            all_change_report = pd.DataFrame()
            woe_df = pd.DataFrame()
            bin_df = pd.DataFrame()

        return all_report, all_change_report, woe_df, bin_df, error_dict

    @staticmethod
    def mannual_rebin(
            data,
            mreport,
            y=None,
            retmr=0,
            lbd=0.5,
            Nan_calculate='all',
            Nan_bin_C_dict={}):
        '''
        Quick method to solve multi-feature
        :param data: DataFrame,the original data
        :param mreport: DataFrame,mannual report
        :param y: the label
        :param retmer: int or float, whether to return the cut
        :param lbd: float,the number which is Dirichlet priori
        :param Nan_calculate: list, the feature which NaN must be caculated
        ,'all' means the Nan of all the feature must be calculated by Nan itself
        ,[] means None of the feature should be calculated by Nan itself
        ,['F1'] means only F1 should be calculated by Nan itself
        :param Nan_bin_C_dict: dict, give the exact the Nan_bin_C
        :return: new bin report
        '''

        data = data.copy()

        if Nan_calculate == 'all':
            Nan_calculate = list(data.columns)

        mreport = mreport.copy()
        if isinstance(mreport['Interval'].iloc[0], str):
            mreport['Interval'] = mreport['Interval'].apply(
                lambda x: json.loads(x))
        mreport = mreport.set_index('Feature')
        all_feature = list(mreport.index.intersection(data.columns))
        if y is not None:
            y = pd.Series(y).copy()
            y.index = data.index
        all_report = pd.DataFrame()
        for feature in all_feature:
            feature_data = data[feature].copy()
            interval = mreport.ix[feature, 'Interval']

            if 'type' in mreport.columns:
                ftype = mreport.ix[feature, 'type']
            else:
                if isinstance(mreport.ix[feature, 'Interval'][0], list):
                    ftype = 'D'
                else:
                    ftype = 'C'

            result, bin_stat, bin_map = Bins.bin_replace(
                feature_data, interval, ftype, y)

            if y is not None:
                Nan_bin_C = {}
                split_bin = []
                for m in range(0, len(bin_stat.index)):
                    interval_tmp = bin_stat['Interval'].iloc[m]
                    if isinstance(interval_tmp, list):
                        if 'NaN' in interval_tmp:
                            if len(interval_tmp) == 1:
                                Nan_bin_C = {
                                    bin_stat['Bin'].iloc[m]: Nan_bin_C_dict.get(
                                        feature, 'max')}
                            split_bin.append(bin_stat['Bin'].iloc[m])
                            break
                    else:
                        split_bin = split_bin + \
                                    list(
                                        bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])
                        Nan_bin_C[list(bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])[
                            0]] = Nan_bin_C_dict.get(feature, 'max')
                        break

                split_bin = list(set(split_bin))

                if feature in Nan_calculate:
                    Nan_bin_C = {}

                woe_result, woeiv, error_flag = Bins.woe_iv(result, y, all_bin=list(
                    bin_stat['Bin']), split_bin=split_bin, lbd=lbd, Nan_bin_C=Nan_bin_C)
                bin_stat = bin_stat.drop(columns=[0, 1, 'total'])
                report = pd.merge(
                    bin_stat, woeiv, left_on='Bin', right_on='Bin')
            else:
                report = bin_stat.copy()

            report['Feature'] = feature
            all_report = all_report.append(report)

        if y is not None:
            all_report = all_report.sort_values(
                by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])

        col_list = list(all_report.columns)
        col_list.remove('Feature')
        all_report = all_report[['Feature'] + col_list]

        mreport = mreport.reindex(all_report.Feature.drop_duplicates())
        mreport = mreport.reset_index()
        mreport['Interval'] = mreport['Interval'].apply(
            lambda x: json.dumps(x))

        if retmr == 0:
            out = all_report
        else:
            out = all_report, mreport

        return out

    @staticmethod
    def whole_bin_replace(data, report):
        '''
        Replace the raw data by bin
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :return: the replaced bin result
        '''

        data = data.copy()
        report = report.copy()

        all_feature = list(
            set(data.columns).intersection(set(report['Feature'])))
        all_result = {}

        for feature in all_feature:
            try:
                if feature in data.columns:
                    feature_report = report[report['Feature']
                                            == feature].copy()
                    feature_data = data[feature].copy()
                    feature_report.index = pd.Series(
                        range(0, len(feature_report.index))).apply(str)
                    feature_report['lower_json'] = feature_report['lower_json'].fillna(
                        'NaN')
                    feature_report['upper_json'] = feature_report['upper_json'].fillna(
                        'NaN')
                    feature_report['lower_json'] = feature_report['lower_json'].apply(
                        lambda x: json.loads(str(x)))
                    feature_report['upper_json'] = feature_report['upper_json'].apply(
                        lambda x: json.loads(str(x)))
                    if feature_report['type'].iloc[0] == 'C':
                        for t in range(0, len(feature_report.index)):
                            if pd.isnull(feature_report.ix[t, 'lower_json']):
                                feature_report.ix[t, 'Interval'] = 'NaN'
                            else:
                                feature_report.ix[t, 'Interval'] = pd.Interval(
                                    feature_report.ix[t, 'lower_json'], feature_report.ix[t, 'upper_json'])
                    else:
                        feature_report['Interval'] = feature_report['lower_json'].copy(
                        )

                    bin_map = Bins.__stat_to_map(
                        feature_report, feature_report['type'].iloc[0])
                    bin_interval = Bins.__stat_to_interval(
                        feature_report, feature_report['type'].iloc[0])
                    if feature_report['type'].iloc[0] == 'C':
                        interval_list = []
                        for i in range(0, len(bin_interval) - 1):
                            interval_list.append(pd.Interval(
                                bin_interval[i], bin_interval[i + 1]))
                        interval_feature_data = pd.cut(
                            feature_data,
                            pd.IntervalIndex(interval_list)).astype(
                            object)
                        interval_feature_data = interval_feature_data.fillna(
                            'NaN')
                        result = interval_feature_data.apply(
                            lambda x: bin_map[x])
                    else:
                        feature_data = feature_data.apply(str)
                        feature_data = Bins.nan_normalize(feature_data)
                        feature_data = feature_data.fillna('NaN')

                        num_len = len(list(bin_map.values())[0]) - 1
                        result = feature_data.apply(
                            lambda x: bin_map.get(
                                x, 'B{:0>{}}'.format(
                                    0, num_len)))
                    all_result[feature] = result
            except Exception as e:
                print(e)
                raise Exception('Feature Error!')

        return pd.DataFrame(all_result)

    @staticmethod
    def whole_woe_replace(data, report, na_fill='max'):
        '''
        Replace the raw data by woe value
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :param na_fill,float or str, if float is the num, if str,only max,min,median is allowed
        :return: DataFrame,the replaced woe result
        '''

        data = data.copy()
        report = report.copy()
        all_feature = list(
            set(data.columns).intersection(set(report['Feature'])))
        bin_data = Bins.whole_bin_replace(data, report)
        all_woe_dict = {}

        for feature in all_feature:
            feature_report = report[report['Feature'] == feature]
            woe_result, woeiv, error_flag = Bins.woe_iv(
                bin_data[feature], report=feature_report)
            na_dict = {
                'max': woe_result.max(),
                'min': woe_result.min(),
                'median': woe_result.median()}
            woe_result = woe_result.fillna(na_dict.get(na_fill, na_fill))
            all_woe_dict[feature] = woe_result

        return pd.DataFrame(all_woe_dict)[bin_data.columns]

    @staticmethod
    def generate_raw_multi(
            data,
            y,
            type_dict={},
            bin_initn_dict={},
            bin_endn_dict={},
            min_bin_prop_dict = {},
            lbd=0.5,
            MDP=False,
            Nan_calculate='all',
            Nan_bin_C_dict={},
            jobn=multiprocessing.cpu_count() *
                 0.8):
        '''
        Generate the report more quickly by multiprocessing
        :param data: DataFrame,the original data
        :param y: list_like,the label
        :param type_dict: dict_like,each type of the feature
        :param bin_initn_dict: dict_like, the init_bin_num of each feature
        :param bin_endn_dict:dict_like, the end_bin_num of each feature
        :param min_bin_prop_dict: dict_like, the min_bin_prop of each feature
        :param lbd: float,the number which is Dirichlet priori
        :param MDP: bool, whether to process the MD data
        :param Nan_calculate: list, the feature which NaN must be caculated
        ,'all' means the Nan of all the feature must be calculated by Nan itself
        ,[] means None of the feature should be calculated by Nan itself
        ,['F1'] means only F1 should be calculated by Nan itself
        :param Nan_bin_C_dict: dict, give the exact the Nan_bin_C
        :param jobn: float, the occupied ratio of the cpus
        :return: tuple, all_report, all_change_report, woe_df, bin_df, error_dict
        '''

        data = data.copy()
        jobn = int(jobn) + 1
        row_s = pd.Series(range(0, len(data.columns)), index=data.columns)
        jobn = min(jobn, len(row_s.index))
        row_cut = pd.qcut(row_s, jobn, labels=range(0, jobn))
        data_list = []
        for i in range(0, jobn):
            data_list.append(data[row_cut[row_cut == i].index])

        mp = multiprocessing.Pool(jobn)
        mplist = []
        for i in range(0, jobn):
            mplist.append(
                mp.apply_async(
                    func=Bins.generate_raw,
                    kwds={
                        'data': data_list[i],
                        'y': y,
                        'type_dict': type_dict,
                        'bin_initn_dict': bin_initn_dict,
                        'bin_endn_dict': bin_endn_dict,
                        'min_bin_prop_dict': min_bin_prop_dict,
                        'lbd': lbd,
                        'MDP': MDP,
                        'Nan_calculate': Nan_calculate,
                        'Nan_bin_C_dict': Nan_bin_C_dict}))

        mp.close()
        mp.join()

        all_report = pd.DataFrame()
        all_change_report = pd.DataFrame()
        woe_df = None
        bin_df = None
        error_dict = {}

        for result in mplist:
            part_report, part_change_report, part_woe_df, part_bin_df, part_error_dict = result.get()
            all_report = all_report.append(part_report)
            all_change_report = all_change_report.append(part_change_report)
            woe_df = pd.concat([woe_df, part_woe_df], axis=1)
            bin_df = pd.concat([bin_df, part_bin_df], axis=1)
            error_dict.update(part_error_dict)

        all_report = all_report.sort_values(
            by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])
        all_change_report = all_change_report.set_index('Feature')
        all_change_report = all_change_report.reindex(
            all_report.Feature.drop_duplicates())
        all_change_report = all_change_report.reset_index()

        return all_report, all_change_report, woe_df, bin_df, error_dict

    @staticmethod
    def mannual_rebin_multi(
            data,
            mreport,
            y=None,
            retmr=0,
            lbd=0.5,
            Nan_calculate='all',
            Nan_bin_C_dict={},
            jobn=multiprocessing.cpu_count() * 0.8
    ):
        '''
        Quicker method to solve multi-feature by multiprocessing
        :param data: DataFrame,the original data
        :param mreport: DataFrame,mannual report
        :param y: the label
        :param retmer: int or float, whether to return the cut
        :param lbd: float,the number which is Dirichlet priori
        :param Nan_calculate: list, the feature which NaN must be caculated
        ,'all' means the Nan of all the feature must be calculated by Nan itself
        ,[] means None of the feature should be calculated by Nan itself
        ,['F1'] means only F1 should be calculated by Nan itself
        :param Nan_bin_C_dict: dict, give the exact the Nan_bin_C
        :param jobn: float, the occupied ratio of the cpus
        :return: new bin report
        '''

        data = data.copy()
        jobn = int(jobn) + 1
        row_s = pd.Series(range(0, len(mreport)), index=mreport['Feature'])
        jobn = min(jobn, len(row_s.index))
        row_cut = pd.qcut(row_s, jobn, labels=range(0, jobn))
        data_list = []
        for i in range(0, jobn):
            data_list.append(data[row_cut[row_cut == i].index])

        mp = multiprocessing.Pool(jobn)
        mplist = []
        for i in range(0, jobn):
            mplist.append(
                mp.apply_async(
                    func=Bins.mannual_rebin,
                    kwds={
                        'data': data_list[i],
                        'mreport': mreport,
                        'y': y,
                        'lbd': lbd,
                        'retmr': 0,
                        'Nan_calculate': Nan_calculate,
                        'Nan_bin_C_dict': Nan_bin_C_dict}))

        mp.close()
        mp.join()

        all_report = pd.DataFrame()

        for result in mplist:
            part_report = result.get()
            all_report = all_report.append(part_report)

        all_report = all_report.sort_values(
            by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])
        mreport = mreport.set_index('Feature')
        mreport = mreport.reindex(all_report.Feature.drop_duplicates())
        mreport = mreport.reset_index()

        if retmr == 0:
            out = all_report
        else:
            out = all_report, mreport

        return out

    @staticmethod
    def whole_bin_replace_multi(
            data,
            report,
            jobn=multiprocessing.cpu_count() *
                 0.8):
        '''
        Replace the raw data by bin more quickly by multiprocessing
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :param jobn: float, the occupied ratio of the cpus
        :return: the replaced bin result
        '''
        data = data.copy()
        report = report.copy()
        jobn = int(jobn) + 1
        all_feature = report['Feature'].drop_duplicates()
        row_s = pd.Series(range(0, len(all_feature)), index=all_feature)
        jobn = min(jobn, len(row_s.index))
        row_cut = pd.qcut(row_s, jobn, labels=range(0, jobn))
        data_list = []
        for i in range(0, jobn):
            data_list.append(data[row_cut[row_cut == i].index])

        mp = multiprocessing.Pool(jobn)
        mplist = []
        for i in range(0, jobn):
            mplist.append(
                mp.apply_async(
                    func=Bins.whole_bin_replace,
                    kwds={
                        'data': data_list[i],
                        'report': report}))

        mp.close()
        mp.join()

        out = pd.DataFrame()

        for result in mplist:
            part_report = result.get()
            out = pd.concat([out, part_report], axis=1)

        return out

    @staticmethod
    def whole_woe_replace_multi(
            data,
            report,
            na_fill='max',
            jobn=multiprocessing.cpu_count() *
                 0.8):
        '''
        Replace the raw data by woe value more quickly by multiprocessing
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :param na_fill,float or str, if float is the num, if str,only max,min,median is allowed
        :param jobn: float, the occupied ratio of the cpus
        :return: the replaced woe result
        '''
        data = data.copy()
        report = report.copy()
        jobn = int(jobn) + 1
        all_feature = report['Feature'].drop_duplicates()
        row_s = pd.Series(range(0, len(all_feature)), index=all_feature)
        jobn = min(jobn, len(row_s.index))
        row_cut = pd.qcut(row_s, jobn, labels=range(0, jobn))
        data_list = []
        for i in range(0, jobn):
            data_list.append(data[row_cut[row_cut == i].index])

        mp = multiprocessing.Pool(jobn)
        mplist = []
        for i in range(0, jobn):
            mplist.append(
                mp.apply_async(
                    func=Bins.whole_woe_replace,
                    kwds={
                        'data': data_list[i],
                        'report': report,
                        'na_fill': na_fill}))

        mp.close()
        mp.join()

        out = pd.DataFrame()

        for result in mplist:
            part_report = result.get()
            out = pd.concat([out, part_report], axis=1)

        return out

    #@staticmethod
    def generate_raw_mono(self, data: pd.DataFrame, y: list, type_dict, MDP=False, jn=0):
        """
        单调分箱
        :param data: 数据集
        :param y: y
        :param all_report:
        :param change_report:
        :return: 同generate_raw
        """
        all_report, change_report, woe_df, bin_df, false_dict = Bins.generate_raw_multi(data, y, type_dict, MDP=MDP)
        var_monoed = list(all_report[all_report["jn"] <= jn]["Feature"].unique())
        var_not_monoed = list(all_report[all_report["jn"] > jn]["Feature"].unique())
        change_report_mono_all = change_report[change_report["Feature"].isin(var_monoed)]

        if len(var_not_monoed) == 0:
            return all_report, change_report, woe_df, bin_df, false_dict
        else:
            while True:
                self.end_num -= 1
                n_limit = self.end_num
                s_data = data[var_not_monoed]
                end_num_dict = {x:self.end_num for x in s_data.columns}
                all_report, change_report, woe_df, bin_df, false_dict = Bins.generate_raw_multi(s_data, y, type_dict,bin_endn_dict = end_num_dict, MDP=MDP)
                print(self.end_num)
                print(change_report.loc[change_report.Feature=='outstand_count'])
                var_monoed = list(all_report[all_report["jn"] <= jn]["Feature"].unique())
                var_not_monoed = list(all_report[all_report["jn"] > jn]["Feature"].unique())
                # change_report_no_mono = all_report[all_report["Feature"].isin(var_not_monoed)]
                change_report_mono = change_report[change_report["Feature"].isin(var_monoed)]
                change_report_mono_all = pd.concat([change_report_mono_all, change_report_mono], axis=0)

                if len(var_not_monoed) == 0:
                    all_report_new = Bins.mannual_rebin_multi(data, change_report_mono_all, y)
                    woe_df_new = Bins.whole_woe_replace_multi(data, all_report_new)
                    return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})
                if n_limit == 2:
                    if len(var_not_monoed) != 0:
                        # print(change_report_mono_all.columns)
                        # print(change_report_no_mono.columns)
                        change_report_no_mono = change_report[change_report["Feature"].isin(var_not_monoed)]
                        change_report_mono_all = pd.concat([change_report_mono_all, change_report_no_mono], axis=0)
                    # print(change_report_mono_all.to_csv("chg_mono.csv"))
                    all_report_new = Bins.mannual_rebin_multi(data, change_report_mono_all, y)
                    woe_df_new = Bins.whole_woe_replace_multi(data, all_report_new)
                    return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})


class StepwiseR(Widget):
    '''
    The class used to do stepwise regression to choose var
    v2.7,copyright by AntiMage-Janhonho,
    Compared to v2.0, I've complete the forward stepwise regression(not validated), the result seems not right
    futher more, I'll complete the forward-backward stepwise,and use more proper evaluation method
    later version will follow by e(2.718281828459045.....)
    '''

    def __init__(self):
        pass

    @staticmethod
    def forward(data, y, logit):
        '''
        forward stepwise LR to choose the best model
        :param data: DataFrame, the raw_data
        :param y: list_like, the label
        :param logit: the sklearn LR regressor
        :return: list,the selected feature
        '''

        x = data.copy()
        y = y.copy()

        remaining = list(data.columns)
        selected = []

        current_score = 0.5
        best_new_score = 0.5

        while len(remaining) > 0 and current_score == best_new_score:
            score_with_candidates = pd.Series(np.nan, index=remaining)
            for candidate in remaining:
                choose = [candidate] + selected
                x_use = x[choose]
                logit.fit(x_use, y)
                ys = pd.Series(clf_l1_lr.predict_proba(x)[:, 1], index=y.index)
                auc_value, ks_value, fpr, tpr, thresholds = Evaluate.cal_auc_ks(
                    y, ys)
                score_with_candidates[candidate] = auc_value

            score_with_candidates = score_with_candidates.sort_values(
                ascending=False)
            best_new_score = score_with_candidates.iloc[0]
            best_candidate = score_with_candidates.index[0]

            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score

        # the realized one seems not the really need one
        raise NotImplementedError

        return selected

    @staticmethod
    def forward_backward(data, y, logit):

        raise NotImplementedError


class Evaluate(Widget):
    '''
    Evaluate Class which can be used to evaluate the Logistic Regression(score_card) result
    v2.7,copyright by AntiMage-Janhonho,
    later version will follow by e(2.718281828459045.....),
    '''

    def __init__(self):
        pass

    @staticmethod
    def cal_auc_ks(y, score):
        '''
        Calculate the auc&ks value
        :param y: list_like,the real result
        :param score: list_like,the score
        :return: tuple of the result
        '''

        if Evaluate.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        fpr, tpr, thresholds = roc_curve(y, score)
        auc_value = auc(fpr, tpr)
        ks_value = max(tpr - fpr)

        return auc_value, ks_value, fpr, tpr, thresholds

    @staticmethod
    def cal_lift(y, score, bins=10):
        '''
        Calculate the lift result
        :param y: list_like,the real result
        :param score: list_like,the score
        :param bins: float or list_like, if float qcut, if list_like cut
        :return: DataFrame, the lift result
        '''

        if Evaluate.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        df = pd.DataFrame({'y': y, 'score': score})
        if isinstance(bins, list):
            if isinstance(bins[0], pd.Interval):
                grouping = pd.cut(df['score'], pd.IntervalIndex(bins))
            else:
                grouping = pd.cut(df['score'], bins)
        else:
            bins = int(bins)
            grouping, retbin = pd.qcut(
                df['score'], bins, duplicates='drop', retbins=True)
            retbin = pd.Series(retbin).apply(lambda x: round(x, 4))
            retbin.iloc[0] = retbin.iloc[0] - 1
            retbin.iloc[-1] = retbin.iloc[-1] + 1
            retbin = retbin.drop_duplicates()
            grouping = pd.cut(df['score'], retbin)

        grouped = df['y'].groupby(grouping)
        df = grouped.apply(
            lambda x: {
                'total': x.count(),
                '1': x.sum()}).unstack()
        df = df.sort_index(ascending=False)

        df = df.reset_index()
        df['0'] = df['total'] - df['1']
        df['1/1_total'] = df['1'] / sum(df['1'])
        df['0/0_total'] = df['0'] / sum(df['0'])
        df['(1/1_total)_cumsum'] = df['1'].cumsum() / sum(df['1'])
        df['(0/0_total)_cumsum'] = df['0'].cumsum() / sum(df['0'])
        df['1/total'] = df['1'] / df['total']
        df['(1/total)_cumsum'] = df['1'].cumsum() / df['total'].cumsum()
        df['ks_score'] = (df['(1/1_total)_cumsum'] -
                          df['(0/0_total)_cumsum']).apply(np.abs)

        df = df[['score', '0', '1', 'total', '1/1_total', '0/0_total',
                 '(1/1_total)_cumsum', '(0/0_total)_cumsum', '1/total',
                 '(1/total)_cumsum', 'ks_score']]

        df = df.sort_values(by='score', ascending=True)

        return df

    @staticmethod
    def cal_confusion_matrix(y, y_predict):
        '''
        Caculate the confusion_matrix
        :param y: list_like,the real y
        :param y_predict: list_like,the predict y
        :return: DataFrame, the confusion matrix
        '''

        if Evaluate.binary_check(
                y) is False or Evaluate.binary_check(y_predict):
            raise ValueError('Value Error of y!')

        result = confusion_matrix(y, y_predict, labels=[0, 1])
        result = pd.DataFrame(result, index=['0', '1'], columns=['0', '1'])

        return result

    @staticmethod
    def get_roc_curve(
            auc_value,
            fpr,
            tpr,
            save_dir=os.getcwd(),
            png_name='ROC_curve.png'):
        '''
        Plot the ROC curve
        :param auc_value: the calculated auc_value
        :param fpr: list_like,the calculated fpr
        :param tpr: list_like,the caculated tpr
        :param save_dir: str, save_dir of the png
        :param png_name: str, the name of the saved pic
        :return:
        '''

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'r--')
        plt.plot(fpr, tpr, label='ROC_curve')
        s = 'AUC:{:.4f}\nKS:{:.4f}'.format(auc_value, max(tpr - fpr))
        plt.text(0.6, 0.2, s, bbox=dict(facecolor='red', alpha=0.5))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC_curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(save_dir, png_name))
        plt.close()

    @staticmethod
    def get_ks_curve(input_df, save_dir=os.getcwd(), png_name='KS_curve.png'):
        '''
        Plot the KS curve
        :param input_df: DataFrame, the caculated ks_df
        :param save_dir: str, save_dir of the png
        :param png_name: str, the name of the saved pic
        :return:
        '''

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = input_df.copy()
        df = df.sort_values(by='score', ascending=False)

        plt.figure()
        plt.plot(
            df['ks_score'].values,
            'r-*',
            label='KS_curve',
            lw=1.2)
        plt.plot(
            df['(0/0_total)_cumsum'].values,
            'g-*',
            label='(0/0_total)_cumsum',
            lw=1.2)
        plt.plot(
            df['(1/1_total)_cumsum'].values,
            'm-*',
            label='(1/1_total)_cumsum',
            lw=1.2)
        plt.plot([0, len(df.index) - 1], [0, 1], linestyle='--',
                 lw=0.8, color='k', label='Random_result')
        xtick = list(df['score'].apply(str))
        plt.xticks(np.arange(len(xtick)), xtick, rotation=60)
        plt.xlabel('Interval')
        plt.ylabel('Rate')
        plt.title('Lift_curve')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, png_name))
        plt.close()

    @staticmethod
    def get_swap():
        '''
        Generate the swap result of the model,but now I've not got a complete idea
        :return:
        '''

        raise NotImplementedError

    @staticmethod
    def prob_to_score(prob, gradient=-28.85, intercept=481.86):
        '''
        Calculate the score from the prob
        :param prob: list_like, the predict probability
        :param gradient: float, the gradient
        :param intercept: float, the intercept
        :return: list_like, the score
        '''
        prob_s = pd.Series(prob).copy()
        score = prob_s.apply(lambda x: gradient *
                                       np.log(x / (1 - x + 1e-7)) + intercept)

        return score

    @staticmethod
    def score_to_prob(score, gradient=-28.85, intercept=481.86):
        '''
        Calculate the prob from the score
        :param score: list_like, the score
        :param gradient: the prob_to_score gradient, the default value -28.85
        :param intercept: the prob_to_score intercept, the default value 481.86
        :return: list_like the prob_list
        '''

        score_s = pd.Series(score).copy()
        tmp_s = score_s.apply(lambda x: np.exp((x - intercept) / gradient))
        prob = tmp_s.apply(lambda x: x / (1 + x))

        return prob

    @staticmethod
    def evaluate(y, score, bins=10, **kwargs):
        '''
        Combined function to calculate all metrics
        :param y: list_like,the real y
        :param score: list_like,the score
        :param bins: float or list_like, if float qcut, if list_like cut
        :param kwargs: dict, the other param,e.g sv_dir,roc_name,ks_name
        :return: dict, all the result
        '''

        gradient = kwargs.get('gradient', -28.85)
        intercept = kwargs.get('intercept', 481.86)
        save_dir = kwargs.get('sv_dir', None)

        auc_value, ks_value, fpr, tpr, thresholds = Evaluate.cal_auc_ks(
            y, score)
        lift_score = Evaluate.prob_to_score(score, gradient, intercept)
        lift_df = Evaluate.cal_lift(y, lift_score, bins)

        if save_dir is not None:
            roc_name = kwargs.get('ROC_name', 'ROC_curve.png')
            Evaluate.get_roc_curve(auc_value, fpr, tpr, save_dir, roc_name)
            ks_name = kwargs.get('KS_name', 'KS_curve.png')
            Evaluate.get_ks_curve(lift_df, save_dir, ks_name)

        result_dict = {'auc_value': auc_value, 'ks_value': ks_value,
                       'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
                       'lift_df': lift_df}

        return result_dict


class Report(Widget):
    '''
    Generate Report for the result
    v2.7,copyright by AntiMage-Janhonho,
    later version will follow by e(2.718281828459045.....),
    '''

    def __init__(self):
        pass

    @staticmethod
    def score(data, SC):
        '''
        use the SC to score the data
        :param data: DataFrame, which include all the feature in the SC
        :param SC: DataFrame, the ScoreCard
        :return: DataFrame, the Score which is by ScoreCard
        '''

        data = data.copy()
        SC = SC.copy()

        SC_feature = pd.Series(SC['Feature'].drop_duplicates())
        if not SC_feature.isin(data.columns).all():
            raise Exception('SC_Error:Feature Loss!')

        if 'Bin' not in SC.columns:
            new_SC = pd.DataFrame()
            for i in list(SC_feature):
                part = SC.ix[SC['Feature'] == i].copy()
                part['Bin'] = range(0, part.shape[0])
                num_len = int(np.log10(len(part.index))) + 2
                part['Bin'] = part['Bin'].apply(
                    lambda x: 'B{:0>{}}'.format(x, num_len))
                new_SC = new_SC.append(part)

            SC = new_SC

        bin_df = Bins.whole_bin_replace(data, SC)
        score_df = bin_df.copy()
        for feature in SC_feature:
            feature_report = SC[SC['Feature'] == feature].copy()
            score_map = feature_report[['Bin', 'Score']].set_index('Bin')[
                'Score'].to_dict()
            score_df[feature] = score_df[feature].apply(
                lambda x: score_map.get(x, np.nan))

        return score_df

    @staticmethod
    def generate_SC(report, coef, intercept, coef_flag=1, path=None):
        '''
        Generate the scorecard sheet
        :param report: Dataframe, the Bins' report
        :param coef: Series, the coef of the LR
        :param intercept: float,the intercept of the LR
        :param coef_flag: float, if 1 means only choose coef!=0
        :return: DataFrame, SC_report
        '''

        SC_report = report.copy()
        coef = coef.copy()
        if coef_flag == 1:
            coef = coef[coef != 0.0]

        try:
            SC_report = SC_report.ix[SC_report['Feature'].isin(coef.index)]
            SC_report['intercept'] = intercept
            SC_report['coef'] = np.nan
            for i in coef.index:
                SC_report.ix[SC_report.Feature == i, 'coef'] = coef.ix[i]

            SC_report['InterScore'] = SC_report['woe'] * \
                                      SC_report['coef'] + 1.0 * intercept / coef.shape[0]
            SC_report['Score'] = -28.85 * \
                                 SC_report['InterScore'] + 481.86 / coef.shape[0]
        except BaseException:
            print('SC_Error')
            SC_report = pd.DataFrame()

        if path is not None:
            sv_dir = os.path.split(path)[0]
            if not os.path.exists(sv_dir):
                os.makedirs(sv_dir)

            SC_report.to_csv(path)

        return SC_report

    @staticmethod
    def generate_SQL(SC, path=None):
        '''
        Generate the sql which is used to score online
        :param SC: DataFrame, the scorecard
        :return: str, the sql(part) which can be used to score the result on database
        '''

        SC = SC.copy()
        SC_feature = pd.Series(SC['Feature'].drop_duplicates())
        sql = ''
        for feature in SC_feature:
            feature_report = SC[SC['Feature'] == feature].copy()
            feature_report.index = pd.Series(
                range(0, len(feature_report.index))).apply(str)
            feature_report['lower_json'] = feature_report['lower_json'].fillna(
                'NaN')
            feature_report['upper_json'] = feature_report['upper_json'].fillna(
                'NaN')
            feature_report['lower_json'] = feature_report['lower_json'].apply(
                lambda x: json.loads(str(x)))
            feature_report['upper_json'] = feature_report['upper_json'].apply(
                lambda x: json.loads(str(x)))

            part_sql = ',(case\r'
            if feature_report['type'].iloc[0] == 'C':
                for t in range(0, len(feature_report.index)):
                    tmp = feature_report.iloc[t]
                    if pd.notnull(tmp['lower_json']):
                        if np.isinf(
                                tmp['lower_json']) and np.isinf(
                            tmp['upper_json']):
                            pass
                        elif np.isinf(tmp['lower_json']):
                            part_sql += 'when {}<={:.4f} then {:.2f}\r'.format(
                                feature, tmp['upper_json'], tmp['Score'])
                        elif np.isinf(tmp['upper_json']):
                            part_sql += 'when {}>{:.4f} then {:.2f}\r'.format(
                                feature, tmp['lower_json'], tmp['Score'])
                        else:
                            part_sql += 'when {}>{:.4f} and {}<={:.4f} then {:.2f}\r'.format(
                                feature, tmp['lower_json'], feature, tmp['upper_json'], tmp['Score'])
                    else:
                        part_sql += 'when {} is null then {:.2f}\r'.format(
                            feature, tmp['Score'])
            else:
                for t in range(0, len(feature_report.index)):
                    tmp = feature_report.iloc[t]
                    dataset = tmp['lower_json']
                    set_sql = ''
                    nan_flag = 0
                    for i in dataset:
                        if i == 'NaN':
                            nan_flag = 1
                        else:
                            set_sql += ',\'{}\''.format(i)
                    set_sql = set_sql[1:]

                    if nan_flag == 1 and set_sql != '':
                        set_sql = '(' + set_sql + ')'
                        part_sql += 'when {} in {} then {:.2f}\r'.format(
                            feature, set_sql, tmp['Score'])
                        part_sql += 'when {} is null then {:.2f}\r'.format(
                            feature, tmp['Score'])
                    elif nan_flag == 1 and set_sql == '':
                        part_sql += 'when {} is null then {:.2f}\r'.format(
                            feature, tmp['Score'])
                    else:
                        set_sql = '(' + set_sql + ')'
                        part_sql += 'when {} in {} then {:.2f}\r'.format(
                            feature, set_sql, tmp['Score'])

            part_sql += 'else null end ) as {}\r'.format(feature)
            sql = sql + part_sql

        sql = sql[1:]
        if path is not None:
            sv_dir = os.path.split(path)[0]
            if not os.path.exists(sv_dir):
                os.makedirs(sv_dir)
            with open(path, 'w') as f:
                f.write(sql)

        return sql

    @staticmethod
    def generate_all_report(
            report,
            coef,
            intercept,
            **kwargs
    ):
        '''
        Generate all the report
        :param report: Dataframe, the Bins' report
        :param coef: Series, the coef of the LR
        :param intercept: float,the intercept of the LR
        :param kwargs: dict, the other param e.g sv_dir,report_name,SC_name,SQL_name,pic_list,lift_dict
        :return DataFrame, the SC_report
        '''

        sv_dir = kwargs.get('sv_dir', os.getcwd())
        report_name = kwargs.get('report_name', 'report.xlsx')
        pic_list = kwargs.get('pic_list', [])
        lift_dict = kwargs.get('lift_dict', {})
        SC_name = kwargs.get('SC_name', 'SC.csv')
        SQL_name = kwargs.get('SQL_name', 'SQL.txt')

        if not os.path.exists(sv_dir):
            os.makedirs(sv_dir)

        writer = pd.ExcelWriter(
            os.path.join(
                sv_dir,
                report_name),
            engine='xlsxwriter')

        summary = pd.DataFrame()
        summary.to_excel(writer, 'Summary')

        SC_report = Report.generate_SC(
            report, coef, intercept, path=os.path.join(
                sv_dir, SC_name))
        SC_report.to_excel(writer, 'Feature')

        book = writer.book
        sheet = book.add_worksheet('AUC&KS')
        x_offset = 0
        y_offset = 0
        n = 0
        for path in pic_list:
            img = mpimg.imread(path)
            sheet.insert_image(
                'B2', path, {
                    'x_offset': x_offset, 'y_offset': y_offset})
            if n % 2 == 1:
                x_offset = 0
                y_offset = y_offset + img.shape[0] + 10
            else:
                x_offset = x_offset + img.shape[1] + 10
            n = n + 1

        n = 0
        for name in lift_dict.keys():
            lift = lift_dict[name].copy()
            lift = lift.rename(columns={'score': name})
            lift.to_excel(writer, 'Lift', startrow=n, index=False)
            n = n + lift.shape[0] + 2

        SC_feature = SC_report.get('Feature', pd.Series()).drop_duplicates()
        SC_feature.index = range(0, len(SC_feature))
        choosen_var = pd.DataFrame({'特征': SC_feature})
        choosen_var['来源'] = np.nan
        choosen_var['备注'] = np.nan
        choosen_var.to_excel(writer, '入选变量及来源')

        if 'Feature' in SC_report.columns:
            SC_simple = SC_report[['Feature', 'Interval', 'Score']].copy()
        else:
            SC_simple = pd.DataFrame()

        SC_simple.to_excel(writer, '评分卡')
        writer.save()

        if SC_report.shape[0] != 0:
            sql = Report.generate_SQL(
                SC_report, os.path.join(
                    sv_dir, SQL_name))

        return SC_report


if __name__ == '__main__':
    root_dir = r'C:\Users\Administrator\Desktop\Bin'
    data = pd.read_csv(os.path.join(root_dir, 'demo_data.csv'))
    x_df = data.drop(columns=['y'])
    x_df = x_df[['F1', 'F2']]
    x_df.ix[0:100, 'F2'] = np.nan
    x_df = x_df.applymap(str)
    y = data['y']

    # 1、快速的生成整个df的结果:

    # all_report是统计特征,
    # change_report是后续用于修改的一个文件，
    # woe_df和bin_df表示按照这样分箱woe替换结果和bin替换结果
    import datetime

    t1 = datetime.datetime.now()
    # all_report, change_report, woe_df, bin_df, false_dict = Bins.generate_raw(
    #     x_df, y, Nan_bin_C_dict={'F1': 'mid'})
    #
    # all_report.to_csv(os.path.join(root_dir, 'report.csv'))
    # change_report.to_csv(os.path.join(root_dir, 'creport.csv'))
    #
    # 通过阅读all_report的结果，已经对creport.csv这个文件进行了修改,后面想看修改后整个结果，如果不满意继续修改creport.csv
    creport = pd.read_csv(os.path.join(root_dir, 'creport.csv'))
    report = Bins.mannual_rebin(x_df, creport, y, Nan_bin_C_dict={'F1': 'min'})
    # report.to_csv(os.path.join(root_dir, 'report.csv'))
    #
    # 如果这个all_report是在之前某次生成的结果
    # report = all_report
    # report = pd.read_csv(os.path.join(root_dir, 'report.csv'))

    # # 对修改的分箱结果满意，那么分别看bin替代结果和woe替代结果
    all_result = Bins.whole_bin_replace(x_df, report)
    all_woe_result = Bins.whole_woe_replace(x_df, report)

    all_woe_result['F3'] = all_woe_result['F1'] + all_woe_result['F2'] + 0.3 * \
                           pd.Series(np.random.randn(len(all_woe_result.index)), index=all_woe_result.index)
    all_woe_result['F4'] = all_woe_result['F1'] + 0.2 * pd.Series(
        np.random.randn(len(all_woe_result.index)), index=all_woe_result.index)

    DataPreprocessing.drop_feature_vif(all_woe_result, threld=10, rmn=2)

    from sklearn.linear_model import LogisticRegression as LR

    clf_l1_lr = LR(penalty='l1')
    clf_l1_lr.fit(all_woe_result, y)
    coef = pd.Series(clf_l1_lr.coef_[0], all_woe_result.columns)

    ys = pd.Series(
        clf_l1_lr.predict_proba(all_woe_result)[
        :, 1], index=y.index)
    r1 = Evaluate.evaluate(y, ys, sv_dir=os.path.join(root_dir, 'train'))

    pic_list = []
    for i in os.listdir(os.path.join(root_dir, 'train')):
        pic_list.append(os.path.join(root_dir, 'train', i))

    Report.generate_all_report(
        report,
        coef,
        clf_l1_lr.intercept_[0],
        pic_list=pic_list,
        lift_dict={
            'train': r1['lift_df']},
        sv_dir=os.path.join(root_dir, 'SCR'))

    SC = Report.generate_SC(
        report,
        coef,
        clf_l1_lr.intercept_[0],
        path=os.path.join(
            root_dir,
            'SCR2',
            'SC.csv'))
    SC = pd.read_csv(os.path.join(root_dir, 'SCR', 'SC.csv'))
    ys = Report.score(x_df, SC)
    Report.generate_SQL(SC)

    # # 2、针对单个特征可以坐以下操作
    #
    # # 2.1 F1是连续变量
    #
    # # 2.1.1 不给出分箱的区间
    # x = data['F1']
    #
    # # 初始化
    # demo = Bins()
    # # 启动运算
    # demo.generate_bin_smi(x, y, ftype='C')
    # # 输出分箱报告
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 2.1.2 给出分箱的区间
    # x = data['F1']
    # demo = Bins()
    # demo.generate_bin_smi(
    #     x, y, interval=[-4.222, -3.186, -2.992, -2.889, -2.883], ftype='C')
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 2.2 F2是离散变量，但是本身有一定顺序，这里顺序可以指定，默认是根据名称
    #
    # # 2.2.1 不给出分箱的区间
    # x = data['F2']
    # demo = Bins()
    # demo.generate_bin_smi(x, y, ftype='M')
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # result = demo.value_to_bin(x)
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 2.2.2 给出分箱的区间
    # x = data['F2']
    # demo = Bins()
    # demo.generate_bin_smi(x, y, interval=[['A01', 'A02']], ftype='M')
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 2.3 F2是离散变量，但是本身不具有顺序
    #
    # # 2.3.1 不给出分箱的区间
    # x = data['F2']
    # demo = Bins()
    # demo.generate_bin_smi(x, y, ftype='D')
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 2.3.2 给出分箱的区间
    # x = data['F2']
    # demo = Bins()
    # demo.generate_bin_smi(x, y, interval=[['A01', 'A02']], ftype='D')
    # bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # # 按照这种分箱将变量替换为箱子值
    # bin_result = demo.value_to_bin(x)
    # # 按照分箱替换的结果计算woe等
    # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # # 按照woe结果将新数据进行woe替换
    # woe_result, woe_report, error_flag = Bins.woe_iv(
    #     bin_result, report=woe_report)
    #
    # # 3 进行简单的替换
    # x = data['F1']
    # bin_result, bin_stat, bin_map = Bins.bin_replace(
    #     x, interval=[-4.222, -3.186, -2.992, -2.889, -2.883], ftype='C')

    print(datetime.datetime.now() - t1)
    print('This is end')
