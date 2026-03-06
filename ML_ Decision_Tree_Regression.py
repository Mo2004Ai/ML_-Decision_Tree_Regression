import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------------------------
#قراء البيانات 

data= pd.read_excel(r"D:\prgrames\program.py\data_seat\House_Rent_Dataset.xlsx")

#----------------------------------------------------------------------
#مرحله فهم البيانات 
#---------------------------------------------------------------------
#المرحله الاوله الفهم العام للبيانات

df=data.copy()  # الخطوه دي انا عملها علشان احافظ البانات الصلايه 

print(df.head())
print (df.columns.to_list)
"""""
['Posted On', 'BHK', 'Rent', 'Size', 'Floor', 'Area Type','Area Locality', 'City', 'Furnishing Status', 
'Tenant Preferred','Bathroom', 'Point of Contact']

BHK..عدد غرف النوم     Floor..دور الشقه 
Rent..سعر الاجار        Area Type.. نوع المساحه
Size..المساحه          Area Locality..اسم المنطقه
City..المدينه          Furnishing Status..حاله الاساس
Tenant Preferred..نوع المستاجر   Bathroom..عدد الحمامات
"""
print(df.shape)#>>>(4746, 12)
print(df.isnull().sum())#>>>Rent /5 , Size /8 , Area Type /7 , Bathroom /4
print(df.nunique())
print(df.info())#>>dtypes: float64(3), int64(1), object(8)

""""
ايه الي استخلصناه من هنا 
1/الاعمده الي فيها قيم غير معرفه هي 
Rent /5 , Size /8 , Area Type /7 , Bathroom /4

2/ Posted On وPoint of Contact الاعمده الي مش هنعوزها هي

"""
df.drop(columns=["Posted On","Point of Contact"],inplace=True)
df.dropna(inplace=True)
print(df.shape)
print(df.info())
#------------------------------------------------------------------------------------
#المرحه الثانيه هي الفهم الاحصاىي

print(df.describe())
"""""
الاستنتاج 
العمودRent فيه قيم شاذه
"""
#هانعمل box plot علشان نتاكد من القيم اشاذه 
col=["BHK","Rent","Size","Bathroom"]
plt.figure(figsize=(10,6))
sns.boxplot(data=df[col])
plt.yscale("log")
plt.title("outlear in data")
plt.show()

""""
الاستنتاج من الرسم ان 
الاعمده الاربعه الرقميه تحتوي علي قيم شاذه ولكن بتفاوت والترتيب كتالي من الاعلي الي الاقل 
 Rent>>Size>>Bathroom>>BHK
 القيم مهمه مش ينفع تتحذف ولكن تتعالج بطرق اخره 
"""
#-----------------------------------------------------------------------------------------------
#مرحله المعالجه من القيم الشاذه
col_log=["BHK_log","Rent_log","Size_log","Bathroom_log"]

df[col_log]=np.log1p(df[col])

#رسم box plot بعد عمل عمل transformation

plt.figure(figsize=(10,6))
sns.boxenplot(data=df[col_log])
plt.title("outlear in data after transformation ")
plt.show()

#----------------------------------------------------------------------------------------------------
#مرحله العلاقات و الرسم 
"""
BHK..عدد غرف النوم int64    Floor..دور الشقه  object
Rent..سعر الاجار float64       Area Type.. نوع المساحه object
Size..المساحه float64         Area Locality..اسم المنطقه object
City..المدينه  object        Furnishing Status..حاله الاساس object
Tenant Preferred..نوع المستاجر  object  Bathroom..عدد الحمامات float64

العلاقات 
1/ سعر الاجار مع كل من عدد الغرف المساحه عدد الحمامات في ضوء المدينه n*n
2/تاثير المكان و المدينه علي السعر n*s
3/تاثير حاله الاساس علي السعر n*s
4/اكثر انواع الاستاجار طلبا one col
"""

fig, axes =plt.subplots(1,3,figsize=(20,6))

sns.scatterplot(data= df , x="Rent",y="BHK",hue="City",ax=axes[0])
axes[0].legend(loc='upper center', bbox_to_anchor=(.9, 1), ncol=2)
axes[0].set_title("Rent vs BHK ")

sns.scatterplot(data= df , x="Rent",y="Size",hue="City",ax=axes[1])
axes[1].legend(loc='upper center', bbox_to_anchor=(.9, 1), ncol=2)
axes[1].set_title("Rent vs Size ")

sns.scatterplot(data= df , x="Rent",y="Bathroom",hue="City",ax=axes[2])
axes[2].legend(loc='upper center', bbox_to_anchor=(.9, 1), ncol=2)
axes[2].set_title("Rent vs Bathroom ")

plt.tight_layout()  
plt.show()

hatmap_povit=df.pivot_table(index="Area Locality" , columns="City",values="Rent")
plt.figure(figsize=(15, 8))
sns.heatmap(hatmap_povit, cmap="YlGnBu", annot=False)
plt.title("Average Rent by City and Area Locality")
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="City", y="Rent", hue="Furnishing Status", ci=None)

plt.title("Average Rent by City and Furnishing Status")
plt.ylabel("Average Rent")
plt.xticks(rotation=45)
plt.legend(title="Furnishing Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Tenant Preferred', palette='pastel')
plt.title("Most Preferred Rental Type (Tenant)")
plt.xlabel("Tenant Type")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------
#مرحله تعليم الاله 
#مرحلة الانكود
""""
دي اعمده جاهزه 
BHK..عدد غرف النوم int64    
Size..المساحه float64        
Bathroom..عدد الحمامات float64

Rent..سعر الاجار float64>>دا الهدف 

الاعمده الي هتحتاج انكودر 
Area Locality..اسم المنطقه object >>LabelEncoder 
City..المدينه  object>>LabelEncoder        
Furnishing Status..حاله الاساس object>>onehotencoder 
Floor..دور الشقه  object>>LabelEncoder 
Tenant Preferred..نوع المستاجر  object>>onehotencoder
Area Type.. نوع المساحه object>>LabelEncoder 
"""
df_ML=df.copy()

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
leab=LabelEncoder()
oneh=OneHotEncoder()

df_ML["Area Locality"]=df_ML["Area Locality"].astype(str)
df_ML["Area Locality"]=leab.fit_transform(df_ML["Area Locality"])

df_ML["City"]=leab.fit_transform(df_ML["City"])

foneh = oneh.fit_transform(df_ML[["Furnishing Status"]])
foneh_df = pd.DataFrame(foneh.toarray(), columns=oneh.get_feature_names_out(["Furnishing Status"]))
foneh_df.index = df_ML.index
df_ML = pd.concat([df_ML.drop("Furnishing Status", axis=1), foneh_df], axis=1)

df_ML["Floor"]=df_ML["Floor"].astype(str)
df_ML["Floor"]=leab.fit_transform(df_ML["Floor"])

tenant_encoded = oneh.fit_transform(df_ML[["Tenant Preferred"]])
tenant_df = pd.DataFrame(tenant_encoded.toarray(), columns=oneh.get_feature_names_out(["Tenant Preferred"]))
tenant_df.index = df_ML.index
df_ML = pd.concat([df_ML.drop("Tenant Preferred", axis=1), tenant_df], axis=1)


df_ML["Area Type"]=leab.fit_transform(df_ML["Area Type"])

print(df_ML.head())

#------------------------------------------------------------------------------------------------------------
#مرحلة السبلت
from sklearn.model_selection import train_test_split

x_trian1= df_ML.drop("Rent", axis=1)
y_train1=df_ML["Rent"]

x_train, x_test, y_train, y_test= train_test_split(x_trian1,y_train1,test_size=0.2,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.3,random_state=42)
print(len(x_train))
print(x_train.shape)
print(len(x_val))
print(x_val.shape)
print(len(x_test))
print(x_test.shape)

#-----------------------------------------------------------------------------------------------------------------
#مرحلة النموذج 

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(max_depth=7,random_state=42)
model.fit(x_train,y_train)

y_pred_test=model.predict(x_test)
y_pred_train=model.predict(x_train)
y_pred_val=model.predict(x_val)

#----------------------------------------------------------------------------------------------------------------
#مرحلة التقيم 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("#############################################<<Training Results>>###################################################")

print("التدريب")

print("MSE:", mse_train)
print("MAE:", mae_train)
print("R² Score:", r2_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test= mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("الاختبار")

print("MSE:", mse_test)
print("MAE:", mae_test)
print("R² Score:", r2_test)

mse_val = mean_squared_error(y_val,y_pred_val)
mae_val= mean_absolute_error(y_val,y_pred_val)
r2_val = r2_score(y_val,y_pred_val)

print("التقيم")


print("MSE:", mse_val)
print("MAE:", mae_val)
print("R² Score:", r2_val)