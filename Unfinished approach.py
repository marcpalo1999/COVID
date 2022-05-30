import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from functions import SIRmodel
import datetime as dt

Question = "d" #input("Answer d for default, otherwise particular")

#Loading data with pandas
data = pd.read_csv('covid_japan.csv')
Data = data.iloc[2:783]

#Generating the Suspicious i Infected columns
N = 125*1e6 #Population of japan
Data['Suspicious'] = Data.loc[:,'Confirmed']-Data.loc[:,'Fatal']-Data.loc[:,'Recovered']
Data['Infected'] = Data.loc[:,'Confirmed']-Data.loc[:,'Fatal']-Data.loc[:,'Recovered']

#Plotting

#Això d'aqui s0ha de fer pl hi ha un bug entre les llibreries pandas y matplotlib per plotejar les dates en l 'eix x
Data['Date']= Data['Date'].astype(str)
Data = Data.set_index('Date')
Data.index = pd.to_datetime(Data.index)


fig, ax = plt.subplots(figsize=(12,12))

ax.plot(Data.index,Data['Infected'], label ='Infected', color = 'orange')
ax.plot(Data.index,Data['Recovered'], label ='Recovered', color = 'yellow')
#ax.plot(Data.index,I1, label ='Prediction Infected', color = 'brown')
ax.legend()
plt.xlabel('Date (day/month/year)', fontsize=12, color='black')
plt.ylabel('Individuals (#)', fontsize=12, color='black')
plt.title('Epidemic data for Japan', fontsize=17, color='green')

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
if Question != "d":
    if input("Want to see the first wave? Write Y for yes otherwise is no.")=="Y":
        left = dt.date(2020,12,1)
        right = dt.date(2021,3,1)
        ax.set_xbound(left,right)
        ax.set_ylim([0,0.6e6])
else:
    a=1
plt.show()
plt.savefig(fname = 'Spain real data')
#print(max(Data.loc[:,'Infected']))
#print(len(Data.loc[:,'Infected']))
##%Y

#Tasques:Y
## Fer la simulació amb SIR i SEIR
## Estimate the infection rate from real data with the RMSE and choose the lower one. Per ferho fem servir el teorema de bolzano.
# Model the effect of q in the evolution
##Compare the suitability of SIR and SEIR to predict the time evolution
## Advanced: Study/Simmulate more realistic infection networks


#We do an optimisation with RMSE evaluation of infection rate r when the recovery rate (a) is 1/2.3 (one parameter optimisation)
#A good method for approaching the best value for r is iterating over shrinking limits

#We choose a date, so we get I0, and we know N = S0*I0*R0. Fiquem que hem tingut 15 dies, i fem un fitting de tots els models per aquells quinze dies. Comprovem si podem prediure el pic de la onada en dia i quantitat i si podem prediure'n el final, en dia i quantitat (I0)






r=1/2.3
beta = 0.5/125e6

#Aquí podriem fer un input
date = "2021-01-01"
I0= Data.loc[Data.index==date,'Infected'][0]
R0= Data.loc[Data.index==date,'Recovered'][0]
S0= N - I0 - R0


cond_inicial=np.array([S0,I0,R0])
t1=np.linspace(1,len(Data.index),len(Data.index)) #The days as a whole



Y=odeint(SIRmodel,           #function
    cond_inicial,  #initial values
    t1,             #time points
    args=(beta,r)) #additional arguments
    #because the function returns an array with the three solutions, we need to separate them
S1=Y[:,0] 
I1=Y[:,1] 
R1=Y[:,2]



plt.figure(figsize = (10,8))
plt.bar(Data.index, I1, color = 'darkcyan', label = "Infected")
plt.xlabel("Time (days)", fontsize = 14)
plt.xticks(fontsize = 12)
plt.ylabel("Italy's Population (#)", fontsize = 14)
plt.yticks(fontsize = 12)
left = dt.date(2020,2,10)
right = dt.date(2021,4,1)
plt.xlim(left,right)
plt.title("Barplot of the COVID-19 INFECTED population in Italy", fontsize = 18)
plt.legend(fontsize = 14)
plt.show()

#Now we are going to approximate the value of beta with an iterative method.
#We'll be working on a specific date, and suppose we have 21 days of data prior to that day. 
#The Infected people at day 0 will be used to produce the different models

r=1/2.3
N=1.25e8

#Preguntem mes dates on fer el fitting
if Question == "d":
    start_date = "2021-01-01"
    final_date = "2021-02-31"
else:
    start_date = input("Input a start date for the data as 2021-01-15")
    final_date = input("Add more or less 21 days to the start")

I0= Data.loc[Data.index==date,'Infected'][0]
R0= Data.loc[Data.index==date,'Recovered'][0]
S0= N - I0 - R0
t0 = np.linspace(0,len(Data.loc[Data.index>date]),len(Data.loc[Data.index>date]))


cond_inicial[0] = Data.loc[start_date,'Suspicious']
cond_inicial[1] = Data.loc[start_date,'Infected']
cond_inicial[2] = Data.loc[start_date,'Recovered']

#Ara el que tractem es de veure pk n dona uns valors logics de beta pel linspace
[min,max]=[-5,2]#Range of values for beta
arg=np.linspace(min,max,10)
betas= [1*10**i for i in arg]
rmse = np.linspace(0,0,10)
betas_final = []
rmse_final = []
iterations = 150
for element in np.linspace(0,iterations,iterations+1):
    

    #Redefinim el vector de betas i reinicialitzem el se rmse
    
    for i in range(0,len(betas)):
        Y=odeint(SIRmodel,           #function
            cond_inicial,  #initial values
            t1,             #time points
            args=(betas[i]/N,r)) #additional arguments
            #because the function returns an array with the three solutions, we need to separate them
        
        mask= (Data.index>start_date)&(Data.index<=final_date)
        Data["I beta "+str(betas[i])]=0
        Data.loc[Data.index>date,"I beta "+str(betas[i])]=Y[len(t0):,1]
        rmse_final.append(np.sqrt(mean_squared_error(Data.loc[mask,'Infected'],Data.loc[mask, "I beta "+str(betas[i])])))
        rmse[i] = np.sqrt(mean_squared_error(Data.loc[mask,'Infected'],Data.loc[mask, "I beta "+str(betas[i])]))

    [betas_final.append(x) for x in betas]
    index_min=np.array(rmse_final).argmin()
    
    [min,max]=[np.log10(betas_final[index_min])-1,np.log10(betas_final[index_min])+1]#Range of values for beta
    arg=np.linspace(min,max,10)
    
    print(index_min,betas)
    print(arg)
    betas= [1*10**i for i in arg]
    


arg = np.array(rmse_final).argmin()
print(betas_final[arg])   

    

plt.figure(figsize = (10,8))
plt.bar(Data.index, Data['Infected'], color = 'darkcyan', label = "Infected")
#plt.bar(Data.index, Data['I beta 0.5'], color = 'darkcyan', label = "Infected")

plt.xlabel("Time (days)", fontsize = 14)
plt.xticks(fontsize = 12)
plt.ylabel("Italy's Population (#)", fontsize = 14)
plt.yticks(fontsize = 12)
left = dt.date(2020,2,10)
right = dt.date(2021,4,1)
plt.xlim(left,right)
plt.title("Barplot of the COVID-19 INFECTED population in Italy", fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
