import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sqlalchemy
from sklearn import metrics
import schedule 
import time
def run():
    engine = sqlalchemy.create_engine("mysql+pymysql://root:12598@localhost:3306/vpulse")
    rawdata = pd.read_sql_table("raw",engine)
##print(rawdata)
##print(rawdata.describe())
    rawcopy = rawdata
    cols = [0,3]
    rawcopy = rawcopy.drop(rawcopy.columns[cols],axis=1)
    rawcopy["count"] = rawcopy.groupby(['consultant','specilization']).transform('count')
    rawcopy['com']=rawcopy['complexity']
    rawcopy.groupby(['consultant','specilization'])
    no_of_cases = rawcopy.groupby(['consultant', 'specilization'], as_index=False).count()
    no_of_cases.to_csv('dash.csv')
    no_of_cases.drop_duplicates(subset ="consultant", keep = False, inplace = True)
    table = pd.pivot_table(rawcopy,index='consultant',columns='complexity',values='com',aggfunc="count").reset_index()
    table.drop_duplicates(subset ="consultant", keep = False, inplace = True)
#print(table)
    cols=[0,2,4]
    no_of_cases.drop(no_of_cases.columns[cols],axis=1,inplace=True)
    combined_data=table.join(no_of_cases)
    copy_CD=combined_data
    copy_CD.rename(columns={'1':'C1','2':'C2'})
    d = {'1':'C1','2':'C2','3':'C3','4':'C4','5':'C5'}
    copy_CD =copy_CD.rename(columns=lambda col: d.get(str(col)) if str(col) in d else col)
    copy_CD = copy_CD[['consultant','specilization','C1','C2','C3','C4','C5','count']]
    copy_CD=copy_CD.fillna(0)
#print(copy_CD)

    copy_CD['efficency'] = 0.2*copy_CD['C1']+0.4*copy_CD['C2']+0.6*copy_CD['C3']+0.8*copy_CD['C4']+1.0*copy_CD['C5']
    copy_CD['Total Effeciency'] = copy_CD['efficency']/copy_CD['count'] 
    copy_CD = copy_CD.rename(columns={'count':'No.of Cases'})
    copy_CD = copy_CD.drop(copy_CD.columns[8],axis=1)
#print(copy_CD)
#copy_CD.to_csv('data.csv')
    comp1 = copy_CD['C1'].values
    comp2 = copy_CD['C2'].values
    comp3 = copy_CD['C3'].values
    comp4 = copy_CD['C4'].values
    comp5 = copy_CD['C5'].values
    case = copy_CD['No.of Cases'].values
    total_effeciency = copy_CD['Total Effeciency'].values
    Cost_Len = len(comp1)
    comp_eff = copy_CD[['C1','C2','C3','C4','C5','No.of Cases']] 
    X = comp_eff.values
    Y = copy_CD.iloc[:, 8].values
#print(X)
#print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.75, test_size = 0.25, random_state = 50)
    reg = linear_model.LinearRegression()
    reg.fit(X_train,Y_train)
#print(reg.intercept_)
#print(reg.coef_)
    y_pred = reg.predict(X_test) ; 
    z_pred = reg.predict(X_train)
#print(y_pred)
#print(z_pred)
    f = pd.DataFrame({'Actual': Y_train, 'Predicted': z_pred})
    r = pd.DataFrame({'Actual': Y_test, 'Predicted':y_pred})
#print(r)
#print(f)
    new_list = list()
    new_list.append(z_pred)
    new_list.append(y_pred)
    Final_list = np.concatenate(new_list)
#print (Final_list)
    s = pd.DataFrame({'consultant_name':copy_CD.consultant, 'specialisation':copy_CD.specilization, 'total_effeciency':Final_list})
#s.to_csv('Doctor_Efffeciency.csv')
    s.to_sql(name='final',
    con=engine,
    if_exists='append',
    index=False)
    print(s)
    print('Root Mean Squared value for test data:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    print('Root Mean Squared value for train data :', np.sqrt(metrics.mean_squared_error(Y_train, z_pred)))
schedule.every(1).minutes.do(run)
while True: 
  
    # Checks whether a scheduled task  
    # is pending to run or not 
    schedule.run_pending() 
    time.sleep(0)