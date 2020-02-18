# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:30:50 2017

@author: Anurag Sharma
"""
import pymysql
import pandas as pd
import datetime as dt
import os
os.environ['TZ'] = 'Asia/Calcutta'
#import matplotlib.pyplot as plt
import xgboost as xg
#from xgboost import XGBRegressor
from sqlalchemy import create_engine
#engine = create_engine('mysql+pymysql://manish:manish123@vps1.climate-connect.com:3306/m_ipt_temp', echo=False)
engine = create_engine('mysql+pymysql://spider:IPTdata123$@35.154.164.151:3306/IndiaPowerTrading', echo=False)

###for weather forecast
def weather_loader(state_id,s_date):
    db_connection = pymysql.connect(host="vps6.climate-connect.com",user="Ashray",password="skyisthelimit",db="Weather")
    SQL1 = "SELECT * FROM w_city WHERE state = '" +state_id+"'"    
    df = pd.read_sql(SQL1, con=db_connection)
    SQL = "SELECT * FROM Weather_historical_india_all WHERE source ='FIO' AND city = '" +df['city'].iloc[0]+"' AND date>='"+str(s_date)+"'"    
    dfa = pd.read_sql(SQL, con=db_connection)
    SQL = "SELECT * FROM Weather_forecasts_india_all WHERE source ='FIO' AND city = '" +df['city'].iloc[0]+"' AND date>'"+str(dfa.loc[len(dfa)-1,'date'])+"' AND date<='"+str(dt.date.today()+dt.timedelta(days=2))+"'" 
    dfb = pd.read_sql(SQL, con=db_connection)
    db_connection.close()
    dfc= dfa.append(dfb)
    del dfa,dfb,SQL1,SQL
    dfc['time']=dfc['time'].apply(lambda x: (dt.datetime.min+ x).time())
    dfc['datetime']=list(map(lambda x,y:pd.datetime.combine(x,y),dfc['date'],dfc['time']))
    dfc=dfc[['datetime','date','time','temperature','dewPoint','humidity','windSpeed','windDirectionDegrees','cloudcover']]
    dfc=dfc[dfc['temperature']>=0]
    dfc = dfc.reset_index(drop=True)
    dist = pd.DataFrame({'datetime':pd.date_range(s_date, dt.date.today()+dt.timedelta(days=2), freq=pd.tseries.offsets.DateOffset(minutes=15))})
    dist=dist[dist.index<len(dist)-1]
    dfc=dist.merge(dfc,on='datetime',how='left')
    dfc['date']=dfc['datetime'].apply(lambda x: x.date())
    dfc['time']=dfc['datetime'].apply(lambda x: x.time())
    dfc['tb']=dfc['datetime'].apply(lambda x: ((x.hour*60+x.minute)//15+1))
    dfc= dfc.interpolate()
    dfc = dfc.reset_index(drop=True)
    dfc=dfc.iloc[:,[0,1,2,9,3,4,5,6,7,8]]
    return(dfc)

###for load
def load(state_id,region):
#    db_connection = pymysql.connect(host="vps1.climate-connect.com",user="manish",password="manish123",db="m_ipt_temp")
    db_connection = pymysql.connect(host="35.154.164.151",user="spider",password="IPTdata123$",db="IndiaPowerTrading")
    if region in ['N1','N2','N3']:
        SQL= "SELECT site_time as datetime, load_value as demand FROM northern_region_state WHERE state = '" +state_id+ "' ORDER BY datetime"
        ddm1 = pd.read_sql(SQL, con=db_connection)
    elif region in ['W1','W2','W3']:
        SQL = "SELECT site_time as datetime,demand FROM wrldc_demand_supply WHERE state = '" +state_id+"' ORDER BY datetime"
        ddm1 = pd.read_sql(SQL, con=db_connection)
    db_connection.close()
    
    ddm1['datetime']=pd.to_datetime(ddm1['datetime'])
    ddm1['datetime'] = ddm1['datetime'].apply(lambda x: dt.datetime(x.year, x.month, x.day, x.hour,15*(x.minute // 15)))
    ddm1=ddm1.sort_values('datetime')
#    ddm1=ddm1[ddm1['datetime'].apply(lambda x: (x.date()>dt.datetime.strptime('27072015', "%d%m%Y").date()))]
#    ddm1=ddm1[ddm1['datetime'].apply(lambda x: (x.date()<=dt.datetime.strptime('08082017', "%d%m%Y").date()))]
    ddm1=ddm1[ddm1['demand']>=0]
    ddm1 = ddm1.reset_index(drop=True)
    ddm1=ddm1.groupby(['datetime'],as_index=False)['demand'].mean()
    qh = ddm1["demand"].quantile(0.9999)
    ql= ddm1["demand"].quantile(0.0001)
    ddm1= ddm1[(ddm1["demand"] < qh) & (ddm1["demand"] > ql)]
    del qh,ql
    ddm1=ddm1.sort_values('datetime')
    ddm1 = ddm1.reset_index(drop=True)   
    ddm1=ddm1.set_index(['datetime'])
    ddm1= ddm1.resample('15T').asfreq()
    ddm1 = ddm1.reset_index()
    ddm1['date']=ddm1['datetime'].apply(lambda x: x.date())
    ddm1['time']=ddm1['datetime'].apply(lambda x: x.time())
    ddm1['tb']=ddm1['datetime'].apply(lambda x: ((x.hour*60+x.minute)//15+1))
    ddm1=ddm1.iloc[:,[0,2,3,4,1]]
    while (any(pd.isnull(ddm1['demand'].head(96)))==True or ddm1.iloc[0,3]!=1):
        if (ddm1.iloc[0,3]!=1 or any(pd.isnull(ddm1['demand'].head(96)))==True):
            ddm1=ddm1[ddm1['date']>ddm1.iloc[0,1]]
            ddm1 = ddm1.reset_index(drop=True)
    dist = pd.DataFrame({'datetime':pd.date_range(ddm1.iloc[0,1], dt.date.today()+dt.timedelta(days=1), freq=pd.tseries.offsets.DateOffset(minutes=15))})
    dist=dist[dist.index<len(dist)-1]
    ddm1=dist.merge(ddm1,on='datetime',how='left')
    ddm1['date']=ddm1['datetime'].apply(lambda x: x.date())
    ddm1['time']=ddm1['datetime'].apply(lambda x: x.time())
    ddm1['tb']=ddm1['datetime'].apply(lambda x: ((x.hour*60+x.minute)//15+1))
    dfList = ddm1['demand'].tolist()
    ddm1['demand']=dfList
    ddp=pd.DataFrame()
    for i in range(1,97):
        dd= ddm1[ddm1['tb']==i]
        dd = dd.reset_index(drop=True)
        dd=dd.set_index(['datetime'])
        dd=dd.interpolate()
        dd['demand'] = dd['demand'].transform(lambda x: x.fillna(x.mean()))
        dd = dd.reset_index()
        ddp=ddp.append(dd)
    del ddm1  
    ddm1=ddp ; del ddp,dd,dfList,i
    ddm1=ddm1.sort_values('datetime')
#    ddm1=ddm1[ddm1['datetime']<dt.datetime.now()]
    ddm1=ddm1.reset_index(drop=True)
    return(ddm1)

def state_names():
    db_connection = pymysql.connect(host="vps6.climate-connect.com",user="Ashray",password="skyisthelimit",db="Weather")
    SQL1 = "SELECT state, region,regionId FROM `w_city` WHERE region in ('N1','N2','N3','W1','W2','W3')"  
    states= pd.read_sql(SQL1, con=db_connection)
    db_connection.close()
    return(states)
state=state_names()
state=state[state.state.isin(['Gujarat','UP','Maharashtra'])]
state = state.reset_index(drop=True)

####for demand
dts1= pd.DataFrame()
me1=dts1
for ii in range(0,len(state)):
    print(state.iloc[ii,0])
    print(state.iloc[ii,1])
    dmd1=load(state.iloc[ii,0],state.iloc[ii,1])

    dw=weather_loader(state.iloc[ii,0],dmd1.iloc[0,1])
    m1= dw.merge(dmd1,on=['datetime','date','time','tb'], how='left')
    del dw
#    plt.plot(m1['datetime'],m1['demand'])
#    plt.plot(m1['datetime'],m1['dewPoint'])
#    plt.plot(m1['datetime'],m1['temperature'])
#    plt.show()
    m1['demand_lag1']= m1['demand'].shift(periods=1)
    m1['demand_lag2']= m1['demand'].shift(periods=2)
    m1=m1[m1['date']>m1.iloc[0,1]]
    m1= m1.reset_index(drop=True)
    m2=pd.DataFrame()
    for k in range(1,97):
        m_temp= m1[m1['tb']==k]
        m_temp= m_temp.reset_index(drop=True)
        m_temp['demand1']= m_temp['demand'].shift(periods=7)
        m_temp['t_demand']= m_temp['demand'].shift(periods=-2)
        m_temp.iloc[:,4:10]=m_temp.iloc[:,4:10].shift(periods=-2)
        m_temp=m_temp[7:]
        m_temp=m_temp[m_temp['date']<=(dmd1.iloc[len(dmd1)-1,1])]
        m2=m2.append(m_temp)
    del k,m_temp
    m2= m2.reset_index(drop=True)
    
    m3=pd.DataFrame()
    for k in range(1,97):
        m_temp= m1[m1['tb']==k]
        m_temp= m_temp.reset_index(drop=True)
        m_temp['demand1']= m_temp['demand'].shift(periods=7)
        m_temp['t_demand']= m_temp['demand'].shift(periods=-3)
        m_temp.iloc[:,4:10]=m_temp.iloc[:,4:10].shift(periods=-3)
        m_temp=m_temp[7:]
        m_temp=m_temp[m_temp['date']<=(dmd1.iloc[len(dmd1)-1,1])]
        m3=m3.append(m_temp)
    del k,m_temp,dmd1,m1
    m3= m3.reset_index(drop=True)
    tt= (dt.datetime.now().hour*60+dt.datetime.now().minute)//15+1
    
    f_date=dt.date.today()-dt.timedelta(days=3)
    t_date=dt.date.today()
#    t_int=pd.Series(pd.date_range(f_date, t_date)).apply(lambda x: x.date())
    t_int= pd.Series([f_date, t_date])
    for i in range(0,len(t_int)):
        dts=me= pd.DataFrame()
        test_date= t_int[i]
        print(test_date)
        for j in range(1,tt):
            train= m2[(m2['tb']==j) & (m2['date']<test_date)]
            train = train.reset_index(drop=True)
            train=train.iloc[:,4:len(m2.columns)]
            
            Dtrain= xg.DMatrix(train.iloc[:,0:(len(train.columns)-1)], label=train.iloc[:,[(len(train.columns)-1)]])
           
            params = {'n_estimators':100,'objective':'reg:linear','booster':'gbtree','max_depth':2,
                      'learning_rate':0.1,'colsample_bytree':0.2}
            model = xg.train(params, Dtrain, num_boost_round=10)    
                        
#            model = XGBRegressor(n_estimators=100,learning_rate=0.1, max_depth=2,colsample_bytree=0.2)
#            model.fit(train.iloc[:,0:(len(train.columns)-1)], train.iloc[:,(len(train.columns)-1)])
            
            test= m2[(m2['tb']==j) & (m2['date']==test_date)]
            test = test.reset_index(drop=True)
            true=test['t_demand'][0]
            test= test.iloc[:,4:(len(test.columns)-1)]
            Dtest = xg.DMatrix(test)
            output = model.predict(Dtest)
            output = output[0]
            error= round(float(abs(output-true)),3)
            error1= round(float((error/true)*100),3)
            dtss= pd.DataFrame({'datetime' : m2[(m2['date']==test_date) & (m2['tb']==j)].iloc[0,0]+dt.timedelta(days=2),'date' : (test_date+dt.timedelta(days=2)),'time_block': j,'demand_A' : true,'demand' : output,'AE': error,'APE': error1,'state': state.iloc[ii,0],'region': state.iloc[ii,1],'region_id': state.iloc[ii,2]},index=[0])
            dtss['time']=dtss['datetime'].apply(lambda x: x.time())
            dtss=dtss.iloc[:,[3,2,10,9,4,5,0,1,8,6,7]]
            if (test_date<dt.date.today()):
                dtss['createTS']= m2[(m2['date']==test_date) & (m2['tb']==tt)].iloc[0,0]+dt.timedelta(days=3)
                dtss['type']='tested'
            elif (test_date==dt.date.today()):
                dtss['createTS']= m2[(m2['date']==test_date) & (m2['tb']==tt)].iloc[0,0]
                dtss['type']='live'
            dts=dts.append(dtss)
            dts= dts.reset_index(drop=True)
            
        test_date= t_int[i]-dt.timedelta(days=1)
        print(test_date)
        
        for j in range(tt,97):
            train= m3[(m3['tb']==j) & (m3['date']<(test_date)-dt.timedelta(days=1))]
            train = train.reset_index(drop=True)
            train=train.iloc[:,4:len(m3.columns)]
            
            Dtrain= xg.DMatrix(train.iloc[:,0:(len(train.columns)-1)], label=train.iloc[:,[(len(train.columns)-1)]])
           
            params = {'n_estimators':100,'objective':'reg:linear','booster':'gbtree','max_depth':2,
                      'learning_rate':0.1,'colsample_bytree':0.2}
            model = xg.train(params, Dtrain, num_boost_round=10) 
        
#            model = XGBRegressor(n_estimators=100,learning_rate=0.1, max_depth=2,colsample_bytree=0.2)
#            model.fit(train.iloc[:,0:(len(train.columns)-1)], train.iloc[:,(len(train.columns)-1)])
        
            test= m3[(m3['tb']==j) & (m3['date']==test_date)]
            test = test.reset_index(drop=True)
            true=test['t_demand'][0]
            test= test.iloc[:,4:(len(test.columns)-1)]
            Dtest = xg.DMatrix(test)
            output = model.predict(Dtest)
            output = output[0]
            error= round(float(abs(output-true)),3)
            error1= round(float((error/true)*100),3)
            dtss= pd.DataFrame({'datetime' : m3[(m3['date']==test_date) & (m3['tb']==j)].iloc[0,0]+dt.timedelta(days=2),'date' : (test_date+dt.timedelta(days=2)),'time_block': j,'demand_A' : true,'demand' : output,'AE': error,'APE': error1,'state': state.iloc[ii,0],'region': state.iloc[ii,1],'region_id': state.iloc[ii,2]},index=[0])
            dtss['time']=dtss['datetime'].apply(lambda x: x.time())
            dtss=dtss.iloc[:,[3,2,10,9,4,5,0,1,8,6,7]]
            if ((test_date)<=dt.date.today()-dt.timedelta(days=3)):
                dtss['createTS']= m3[(m3['date']==test_date) & (m3['tb']==tt)].iloc[0,0]+dt.timedelta(days=4)
                dtss['type']='tested'
            elif (test_date==(dt.date.today()-dt.timedelta(days=1))):
                dtss['createTS']= m3[(m3['date']==test_date) & (m3['tb']==tt)].iloc[0,0]+dt.timedelta(days=1)
                dtss['type']='live'
            dts=dts.append(dtss)
            dts= dts.reset_index(drop=True)
        dts.to_sql(name='Load_IPT', con=engine, if_exists = 'append', index=False)
        if (i==0):
            MAE= round(float(dts.loc[(dts['date']==(test_date+dt.timedelta(days=3))),'AE'].mean()),3)
            MAPE= round(float(dts.loc[(dts['date']==(test_date+dt.timedelta(days=3))),'APE'].mean()),3)
            print(MAE)
            print(MAPE)
            me=pd.DataFrame({'date' : (test_date+dt.timedelta(days=3)), 'MAE' : [MAE],'MAPE': [MAPE],'state': state.iloc[ii,0],'region': state.iloc[ii,1],'region_id': state.iloc[ii,2]})
            me=me.iloc[:,[2,0,1,5,3,4]]
            me['createTS']=dtss['createTS'].iloc[0]
            me['createTS']= m2[(m2['date']==test_date+dt.timedelta(days=3)) & (m2['tb']==tt)].iloc[0,0]
            me.to_sql(name='Load_IPT_acc', con=engine, if_exists = 'append', index=False)
            me1=me1.append(me)
            me1=me1.reset_index(drop=True)
            del me,MAE,MAPE
        dts1=dts1.append(dts)
        del output,true,train,test,error,error1,dtss,i,j,model
    del dts,test_date
del ii,m2,m3,t_date,f_date,t_int
dts1=dts1.reset_index(drop=True)
