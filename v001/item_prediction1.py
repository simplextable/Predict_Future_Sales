import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML
from scipy.stats import norm
# collection of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
pd.set_option("display.max_rows",22222200)  # KISALTMA ENGELLEME
pd.set_option("display.max_rows",2222200)  # KISALTMA ENGELLEME


ss= pd.read_csv('sample.csv')
item_cat = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
sales_train = pd.read_csv('sales_train.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')



nihai = pd.merge(sales_train, test, how='left', on=['shop_id', 'item_id'])
nihai2 = pd.merge(sales_train, items, how='left', on=['item_id'])
print(nihai.isnull().sum())
nihai4 = pd.merge(sales_train, test, how='right', on=['shop_id', 'item_id'])
print(nihai4.isnull().sum())
#######    Ürün 33 ayda ne kadar satılmış
#NOT : Ürün satım sayısı düşük

aylik2 = nihai.groupby(['ID',"item_id","shop_id"])['item_cnt_day'].agg("sum").reset_index()
aylik2 = aylik2.rename(columns={'item_cnt_day': 'toplam_satis'})
aylik3 = nihai.groupby(['ID',"item_id","shop_id"])['date_block_num'].agg("min").reset_index()
aylik3 = aylik3.rename(columns={'date_block_num': 'min_ay'})
aylik = nihai.groupby(['ID',"item_id","shop_id"])['date_block_num'].agg("max").reset_index()
aylik = aylik.rename(columns={'date_block_num': 'max_ay'})
aylik4 = pd.concat([aylik,aylik2["toplam_satis"],aylik3["min_ay"]], axis=1)

aylik4["reyon_suresi"] = 33

for index, i in aylik4.iterrows():
        aylik4["reyon_suresi"][index]= aylik4["max_ay"][index] - aylik4["min_ay"][index] +1

for index, i in aylik4.iterrows():
        aylik4["toplam_satis"][index]= aylik4["toplam_satis"][index]/aylik4["reyon_suresi"][index]  

aylik4 = aylik4.rename(columns={'toplam_satis': 'ortalama_aylik_satis'})

aylik5 = pd.merge(aylik4, items, how='left', on=['item_id'])

aylik5.drop(['item_name', "max_ay", "min_ay", "reyon_suresi", ],axis=1,inplace=True)
aylik6 = aylik5[["shop_id","item_category_id"]]
aylik_target = aylik5[["ortalama_aylik_satis"]]

aylik7 = pd.concat([aylik6,test_rf], axis=0)

aylik8 = pd.get_dummies(aylik7,columns=["shop_id", "item_category_id"]).reset_index(drop=True)

aylik9 = aylik8.iloc[:aylik6.shape[0], :]
aylik_test = aylik8.iloc[aylik6.shape[0]:, :]


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylik6,aylik_target,test_size=0.33, random_state=0)


#aylik = aylik.rename(columns={'item_cnt_day': 'toplam_month'})
 
########################################

aylik_cat_ekln = pd.merge(aylik, items, how='left', on=['item_id'])
aylik_cat_ekln.drop(['item_name'],axis=1,inplace=True)
aylik_cat_ekln["reyon_suresi"] = 33


for index, i in aylik_cat_ekln.iterrows():
        aylik_cat_ekln["toplam_month"][index]= aylik_cat_ekln["toplam_month"][index]/aylik_cat_ekln["reyon_suresi"][index]  

aylik_cat_ekln2 = aylik_cat_ekln[["toplam_month"]] / 33
aylik_cat_ekln2 = aylik_cat_ekln2.rename(columns={'toplam_month': 'item_cnt_month'})
final_aylik=pd.concat([aylik_cat_ekln,aylik_cat_ekln2],axis=1)



########################################

final_aylik_rf = final_aylik[["shop_id","item_category_id"]]
final_aylik_rf_target = final_aylik[["item_cnt_month"]]


test_cat = pd.merge(test, items, how='left', on=['item_id'])
test_rf = test_cat[["shop_id","item_category_id"]]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(final_aylik_rf,final_aylik_rf_target,test_size=0.33, random_state=0)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(x_train,y_train)
y_pred = rf_reg.predict(test_rf)
pred_df = pd.DataFrame(data = y_pred , index=range(214200), columns=['item_cnt_month'])







ara_veri = final_aylik[["ID","shop_id","item_id","item_category_id","item_cnt_month"]]
pred_nihai=pd.concat([test["ID"],pred_df],axis=1)

merge_pred =  pd.merge(pred_nihai, ara_veri, how='left', on=['ID'])

merge_pred.drop(["shop_id","item_id","item_category_id"],axis=1,inplace=True)

print(merge_pred["item_cnt_month_x"][0])
for index, i in merge_pred.iterrows():
    if (pd.isnull(i["item_cnt_month_y"])):
        merge_pred["item_cnt_month_y"][index]= merge_pred["item_cnt_month_x"][index] 
        
        
merge_pred.drop(["item_cnt_month_x"],axis=1,inplace=True)
merge_pred = merge_pred.rename(columns={'item_cnt_month_y': 'item_cnt_month'})
pred_nihai.to_csv('Ha-Al_11_05_2.csv',index=False)



        
        