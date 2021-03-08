import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  LabelEncoder,StandardScaler,MinMaxScaler, Normalizer

  '''

  @author : Emirhan KIRAN
  @date : 08/03/2021
  @project_name : Dataset Explore Tool
  @licence : none

  '''


class Preprocessing:

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        
    def explore_dataset(self):
        '''
        Bu fonksiyon veri seti hakkında genel bilgiler vermektedir.
        '''
        print(f'Veri setinde {self.dataset.shape[0]} satır veri bulunmakta.')
        print(f'Veri setinde {self.dataset.shape[1]} adet öznitelik bulunmakta.\n')

        print(self.dataset.head(3))

        categorical = self.dataset.select_dtypes(exclude=['int','float'])
        numeric = self.dataset.select_dtypes(include=['int','float'])
        number_but_categorical = [cols for cols in self.dataset.columns if self.dataset[cols].nunique() > 10 and cols in categorical and cols not in numeric]
        categorical = categorical.drop(labels=number_but_categorical,axis=1)
        print(f'\n\nVeri setinde {len(categorical.columns)} adet kategorik öznitelik bulunmaktadır.')
        print('--------------------------------')
        [print(index+1,' --> ',item) for index,item in enumerate(categorical.columns)]
        [print(f'\nUYARI! >>> {item} özniteliğinin sahip olduğu eşsiz değer sayısı : {categorical[item].nunique()}\nBu özniteliği one hot encoding ile dönüştürebilirsiniz.') for item in categorical.columns if categorical[item].nunique() > 2]
        print('--------------------------------')
        print(f'\nVeri setinde {len(numeric.columns)} adet nümerik (sayısal) öznitelik bulunmaktadır.')
        print('--------------------------------')
        [print(index+1,' --> ',item) for index,item in enumerate(numeric.columns)]
        print(f'\nVeri setinde {len(number_but_categorical)} adet nümerik fakat kategorik olarak belirtilmiş öznitelik var.')
        print('--------------------------------')
        [print(cols) for cols in number_but_categorical]
        print('--------------------------------\n\n')

        print('Veri setinde bulunan kolonların isimleri : ')
        print('--------------------------------')
        [print(index+1,' --> ',item) for index,item in enumerate(self.dataset.columns)]
        print('--------------------------------')

        print('Veri setinize ait boş (None) değerlerin dağılımı')
        print('--------------------------------')
        print(self.dataset.isnull().sum())
        print('--------------------------------')
        print(f'Toplam boş değerlerin sayısı : {self.dataset.isnull().sum().sum()}\n\n')

        print('Kategorik özniteliklerin değer dağılımları : ')
        [print(f'--------------------------------\n{self.dataset[item].value_counts()}') for item in categorical.columns]
        print('--------------------------------\n\n')

        numeric.describe(percentiles=[0.10,0.25,0.35,0.50,0.75,0.85,0.90,1]).T

        # Outlier tespiti
        for item in numeric:
            Q1 = list(numeric[item].quantile([0.25]))[0]
            Q3 = list(numeric[item].quantile([0.75]))[0]
            IQR = Q3-Q1
            min_val = Q1 - 1.5*IQR
            max_val = Q3 + 1.5*IQR
            outlier_values = [data for data in numeric[item].values if (data > max_val) and (data < min_val)]
            print(f'{item} özniteliği için aykırı değerlerin sayısı : {len(outlier_values)}')
            [print(data) for data in outlier_values if len(outlier_values) > 0]
        print('\n\nNumerik verilere ait istatistiksel veriler : \n\n')
        print(self.dataset.describe().T)
    

    def dependent_explore(self):

        '''
        Bu fonksiyon bağımlı değişken için bir dağılım grafiği ve kutu grafiği çizmektedir.
        Bu sayede verilerinizin nasıl dağıldığını ve aykırı değerlerin olup olmadığını görebilirsiniz. 
        '''

        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1)

        plt.title(self.target + " Distribution Plot")
        sns.distplot(self.dataset[self.target])

        plt.subplot(1,2,2)
        plt.title(self.target + " Spread")
        sns.boxplot(y=self.dataset[self.target])

        plt.show()

    def explain_column(self,column_name):

        '''
        column_name : İstatistiksel verilerini görmek istediğiniz özniteliğin adı.

        Bu metod ile seçtiğiniz özniteliğe ait dağılım sayılarını, her değere
        ait dağılım sayısını ve medyan değerini görürüz.

        Örnek kullanım : explain_column('gender','price')
        '''
        print(f'\tKolon : {column_name}\n')
        print(pd.DataFrame({'ADET':self.dataset[column_name].value_counts(),
                            'ORAN':self.dataset[column_name].value_counts()/len(self.dataset),
                            'MEDYAN ':self.dataset.groupby(column_name)[self.target].mean()}).sort_values(by='ADET', ascending=False))


    def get_columns(self):
        
        '''
        Bu metod vermiş olduğunuz veri setindeki öznitelikleri kategorik, nümerik ve
        nümerik fakat kategorik olarak ayırarak size geri döndürür. 
        '''

        categorical = [col for col in self.dataset.columns if self.dataset[col].dtype == 'O']
        numeric = [col for col in self.dataset.columns if self.dataset[col].dtype != 'O' and self.dataset[col].nunique() > 5]
        numeric_but_categorical = [col for col in self.dataset.columns if col not in categorical and col not in numeric]
        
        return categorical, numeric, numeric_but_categorical


    def get_one_hot_encoding(self,drop_first = False):

        '''
        drop_first = Bu parametre ile elde ettiğiniz kolonlardan ilkinin düşürülmesini
        sağlayabilirsiniz. Bunun amacı dummy varaibleslardan (kukla değişkenler)
        kurtulmaktır. 

        Bu metod veri setinizde bulunan kategorik verilerden, değer sayısı ikiden
        fazla olan değişkenler için one hot encoding yöntemini uygulayarak size
        encode edilmiş bir DataFrame nesnesi döndürür. Ön tanımlı olarak 
        drop_first parametresi ' FALSE ' olarak gelmektedir.

        Örnek kullanım : get_one_hot_encoding(drop_first = True)
        '''

        categorical_columns = [column for column in self.dataset.columns if self.dataset[column].nunique() > 2 and self.dataset[column].nunique() < 10 and  self.dataset[column].dtype == 'O']
        ohe_columns = pd.get_dummies(self.dataset[categorical_columns],drop_first=drop_first)
        return ohe_columns

    def get_label_encoding(self):

        '''
        Bu metod veri setinizde bulunan kategorik verilerden, değer sayısı ikiye
        eşit olan değişkenler için label encoding yöntemini uygulayarak size
        encode edilmiş bir DataFrame nesnesi döndürür.

        Örnek kullanım : get_label_encoding()
        '''

        #categorical_columns = self.dataset.select_dtypes(exclude=['int','float'])
        le_columns = [column for column in self.dataset.columns if self.dataset[column].nunique() == 2 and self.dataset[column].dtype == 'O']
        le = LabelEncoder()
        le_cols = self.dataset[le_columns].apply(le.fit_transform)
        return le_cols

    def concatenate_columns(self,**kwargs):
        '''
        **kwargs = anahtar argüman olarak birleştirmek istediğiniz DataFrame'leri alır.
        firt=df şeklinde ilk parametreyi verebilirsiniz.

        Bu metod parametre olarak verdiğiniz tüm DataFrame nesnelerini birleştirir ve
        geriye bir DataFrame nesnesi döndürür.

        Örnek kullanım : concatenate_columns(first=numeric,second=cat)
        '''
        dataframe = pd.DataFrame()
        for item in kwargs:
            dataframe=pd.concat([dataframe,kwargs.get(item)],axis=1)
        return dataframe
    
    def normalize_data(self,target_column):
        '''
        target_column : Normalizasyon işleminin uygulanacağı öznteliğin ismini temsil
        eder. Pandas Series tipinde girdi alır. 

        Bu metod seçmiş olduğunuz öznitelikteki verileri ölçekleyerek normal dağılıma
        dönüştüren bir yapıdır. Bu sayede daha performanslı modeller elde edebilebilir
        Geriye bir array nesnesi döner.

        Bu değerleri DataFrame'e eklemek için aşağıdaki örnek koda benzer bir kod
        yazabilirsiniz.

        ==> dataFrame['temperature'] = normalize_data(dataFrame['temperature']).reshape(len,1)

        Örnek kullanım : normalize_data('price')
        '''
        normalizer = Normalizer()
        normalizer.fit([target_column])
        return normalizer.transform([target_column])
