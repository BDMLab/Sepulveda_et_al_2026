# import pymc as pm
import pandas as pd
#import theano.tensor as tt
#import theano as theano
import arviz as az
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import seaborn as sns
from sklearn import linear_model  # packages for the logistic regression function to plot the logistic regression 
from sklearn.linear_model import LogisticRegression # 
from matplotlib import cm
import statsmodels.formula.api as sm



def logisticplot_simplDots (modlow, data, xaxis='zDV', yaxis='G_choice', ylab='P(Chose Reference Item)', xlab='DV (Z-score)',
                  meanCol='#AAAAAA',subjCol='#000000', title='empty', xlim = [-5,5]):
    
    sns.set(font_scale=1.5, style='white')
    figsize(5,5)
    
    # defining the sigmoid function
    def model(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    sub = plt.subplot()


    #run the classifier
    clf = linear_model.LogisticRegression(C=1e5)

    logit_low = {}

    # I think this defines the problem space
    X_test = np.linspace(-10,10,300)

    # fitting the predictive logistic model for the low_confidence trials, for a participant specified by x
    # first I specify the value difference right - left, then I specify the choices, left or right
    clf.fit(np.array(data[xaxis])[:, np.newaxis],
            data [yaxis])
    logit_low = model(X_test*clf.coef_ + clf.intercept_).ravel()
    print ('Low measure coef',clf.coef_)
    
    #      data.groupby(['Part', 'difficulty']).choice.mean()

    
    # generate scatter to plot pariticipants means
    
    levels =  (np.max(data[xaxis]) - np.min(data[xaxis]))/10
    lev_label = np.arange(np.min(data[xaxis]), np.max(data[xaxis]) + levels,levels) 
    
    difficulty2= []
    for i in range(len(data[xaxis].values)):
         difficulty2.append( lev_label[ int((data[xaxis].values[i] - np.min(data[xaxis]) )//levels)] )
            
    data['difficulty'] = np.around(difficulty2, decimals = 3)        
    
    subject_means = data.groupby(['Part', 'difficulty']).choice.mean()
    means = subject_means.groupby('difficulty').mean()
    sems = subject_means.groupby('difficulty').sem()
    
    #Plotting the predictive lines
    #line_low = sub.scatter(X_test, logit_low, color=modlowcol, linewidth=5, label=modlow, zorder=5) 

    # plot subject means
    scatter_data = subject_means.reset_index()
    x_scatter = scatter_data['difficulty'] 
    jittr = np.random.uniform(low=-max(x_scatter)/10,high=max(x_scatter)/10,size=len(scatter_data))/2
    sub.plot(x_scatter+jittr, scatter_data.choice.values, marker='o', ms=5, markerfacecolor=subjCol, color=subjCol,alpha=0.3,linestyle="None")

    # plot mean values
    sub.plot(list(means.index), means.values, 'o', markerfacecolor=meanCol, markersize = 10, fillstyle = 'full',
                    color=meanCol, linewidth=1,label=modlow)
    sub.vlines(list(means.index), means - sems, means + sems,
                      linewidth=1, color= meanCol)
    
    # Set Labels
    sub.set_ylabel(ylab, fontsize=30)
    sub.set_xlabel(xlab, fontsize=30)

    # Set Ticks
    #sub.set_xticks((-5,-3,-1,1,3,5))
    sub.set_yticks((0,0.25,0.5,0.75,1))
    sub.tick_params(axis='both', which='major', labelsize=20)

    # Set Limits
    sub.set_ylim(-0.01, 1.01)
    sub.set_xlim(xlim[0], xlim[1])

    # Set Title
    if title == 'empty':
        sub.set_title('')
    else:
        sub.set_title(title)
    
    sub.legend(frameon=False, prop={'size':20})
    
    sns.despine()



def logisticplot_simpl (modlow, data, xaxis='zDV', yaxis='G_choice', ylab='P(Chose Reference Item)', xlab='DV (Z-score)',
                  modlowcol='#AAAAAA', title='empty', xlim = [-5,5],linewidth = 5, x_test_lim = 10):
    
    sns.set(font_scale=1.5, style='white')
    figsize(5,5)
    
    # defining the sigmoid function
    def model(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    sub = plt.subplot()


    #run the classifier
    clf = linear_model.LogisticRegression(C=1e5)

    logit_low = {}

    # I think this defines the problem space
    X_test = np.linspace(-x_test_lim,x_test_lim,300)

    # fitting the predictive logistic model for the low_confidence trials, for a participant specified by x
    # first I specify the value difference right - left, then I specify the choices, left or right
    clf.fit(np.array(data[xaxis])[:, np.newaxis],
            data [yaxis])
    logit_low = model(X_test*clf.coef_ + clf.intercept_).ravel()
    print ('Low measure coef',clf.coef_)
    
    #Plotting the predictive lines
    line_low = sub.plot(X_test, logit_low, color=modlowcol, linewidth=5, label=modlow, zorder=5) 
    
    # Set Labels
    sub.set_ylabel(ylab, fontsize=30)
    sub.set_xlabel(xlab, fontsize=30)

    # Set Ticks
    sub.set_xticks((-5,-3,-1,1,3,5))
    sub.set_yticks((0,0.25,0.5,0.75,1))
    sub.tick_params(axis='both', which='major', labelsize=20)

    # Set Limits
    sub.set_ylim(-0.01, 1.01)
    sub.set_xlim(xlim[0], xlim[1])

    # Set Title
    if title == 'empty':
        sub.set_title('')
    else:
        sub.set_title(title)
    
    sub.legend(frameon=False, prop={'size':20})
    
    sns.despine()


def logisticplot_LowHigh (moderator, modhigh, modlow, data, xaxis='zDV', yaxis='G_choice', ylab='P(Chose Reference Item)', xlab='DV (Z-score)',
                 modhighcol='#000000', modlowcol='#AAAAAA', title='empty',x_test_lim = 10, xlim = [-5,5]):
    
    sns.set(font_scale=1.5, style='white')
    
    figsize(5, 5)

    
    # defining the sigmoid function
    def model(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    sub = plt.subplot()


    #run the classifier
    clf = linear_model.LogisticRegression(C=1e5)

    # Paula used these dictionaries to store the values of the predictive lines for all the participants.
    logit_low = {}
    logit_high = {}

    # I think this defines the problem space
    X_test = np.linspace(-x_test_lim,x_test_lim,300)

    

    # fitting the predictive logistic model for the low_confidence trials, for a participant specified by x
    # first I specify the value difference right - left, then I specify the choices, left or right
    clf.fit(np.array(data.loc[data[(data[moderator]==0)].index, xaxis])[:, np.newaxis],
            data.loc[data[(data[moderator]==0)].index, yaxis])
    logit_low = model(X_test*clf.coef_ + clf.intercept_).ravel()
    print ('Low measure coef',clf.coef_)
    
    # fitting the predictive logistic model for the high_confidence trials, for a participant specified by x
    # first I specify the value difference right - left, then I specify the choices, left or right
    clf.fit(np.array(data.loc[data[(data[moderator]==1)].index, xaxis])[:, np.newaxis],
            data.loc[data[(data[moderator]==1)].index, yaxis])
    logit_high = model(X_test * clf.coef_ + clf.intercept_).ravel()
    print ('High measure coef',clf.coef_)



    #Plotting the predictive lines
    line_high = sub.plot(X_test, logit_high, color=modhighcol, linewidth=5, label=modhigh, zorder=6)
    line_low = sub.plot(X_test, logit_low, color=modlowcol, linewidth=5, label=modlow, zorder=5) 
    
    # Set Labels
    sub.set_ylabel(ylab, fontsize=30)
    sub.set_xlabel(xlab, fontsize=30)

    # Set Ticks
    sub.set_xticks((-5,-3,-1,1,3,5))
    sub.set_yticks((0,0.25,0.5,0.75,1))
    sub.tick_params(axis='both', which='major', labelsize=20)

    # Set Limits
    sub.set_ylim(-0.01, 1.01)
    sub.set_xlim(xlim[0], xlim[1])

    # Set Title
    if title == 'empty':
        sub.set_title('')
    else:
        sub.set_title(title)
    
    sub.legend(frameon=False, prop={'size':20})
    #leg.get_frame().set_linewidth(0.0)

    sns.despine()


def logisticplot_simpl_sims (modlow, data, xaxis='zDV', yaxis='G_choice', ylab='P(Chose Reference Item)', xlab='DV (Z-score)',
                  modlowcol='#AAAAAA', title='empty',xlim = [-5,5],loc_legend = 0,x_test_lim = 5, legend_on = False, sub = plt.subplot()):
    
   # sns.set(font_scale=1.5, style='white')
   # figsize(5,5)
    
    # defining the sigmoid function
    def model(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
   # sub = plt.subplot()


    #run the classifier
    clf = linear_model.LogisticRegression(C=1e5)

    logit_low = {}

    # I think this defines the problem space
    X_test = np.linspace(-x_test_lim,x_test_lim,300)

    # fitting the predictive logistic model for the low_confidence trials, for a participant specified by x
    # first I specify the value difference right - left, then I specify the choices, left or right
    clf.fit(np.array(data[xaxis])[:, np.newaxis],
            data [yaxis])
    logit_low = model(X_test*clf.coef_ + clf.intercept_).ravel()
    print ('Low measure coef',clf.coef_)
    
    #Plotting the predictive lines
    line_low = sub.plot(X_test, logit_low, color=modlowcol, linewidth=5, label=modlow, zorder=5) 
    
    # Set Labels
    sub.set_ylabel(ylab, fontsize=30)
    sub.set_xlabel(xlab, fontsize=30)

    # Set Ticks
    #sub.set_xticks((-5,-3,-1,1,3,5))
    sub.set_yticks((0,0.25,0.5,0.75,1))
    sub.tick_params(axis='both', which='major', labelsize=20)

    # Set Limits
    sub.set_ylim(-0.01, 1.01)
    sub.set_xlim(xlim[0], xlim[1])
    # Set Title
    if title == 'empty':
        sub.set_title('')
    else:
        sub.set_title(title)
    
    if legend_on:
        sub.legend(loc=loc_legend, prop={'size':18}, frameon = False)
    
    sns.despine()




def zscore1(data_all,z_score_var, part_def = 0):
    z_matrix=[]
    z_matrix_aux=[]
    
    if part_def==0:
        z_matrix = (data_all[z_score_var] - data_all[z_score_var].mean())/ data_all[z_score_var].std()
    else:
        for i in (data_all[part_def].unique()):
            Choicedata = data_all.loc[data_all[part_def] == i]    
        
            pX_A= pd.to_numeric(Choicedata[z_score_var]) 
            pX_zA= (pX_A - np.mean(pX_A))/np.std(pX_A)
    
            z_matrix_aux= pX_zA.values
        
            for  j in range(len(z_matrix_aux)):    
                z_matrix.append(z_matrix_aux[j])
    return z_matrix


def norm01(data_all,z_score_var, part_def = 0):
    z_matrix=[]
    z_matrix_aux=[]
    
    if part_def==0:
        z_matrix = (data_all[z_score_var] - data_all[z_score_var].min())/ (data_all[z_score_var].max() - data_all[z_score_var].min())
    else:
        for i in (data_all[part_def].unique()):
            Choicedata = data_all.loc[data_all[part_def] == i]    
        
            pX_A= pd.to_numeric(Choicedata[z_score_var]) 
            pX_zA= (pX_A - np.min(pX_A))/(np.max(pX_A) - np.min(pX_A))
    
            z_matrix_aux= pX_zA.values
        
            for  j in range(len(z_matrix_aux)):    
                z_matrix.append(z_matrix_aux[j])
    return z_matrix


def split_values_highLow (value_sort):

    value_sort.sort()
    value_low = value_sort[:int(len(value_sort)/2)]
    value_high = value_sort[int(len(value_sort)/2) + 1 :]

    plt.title(r"Distribution of low and high values")
    figsize(7, 7)
    plt.hist(value_low, bins=10, alpha=0.85,
                label=r"Low Value", color="#7A68A6")#, normed=True)
    plt.hist(value_high, bins=10, alpha=0.85,
                label=r"High Value", color="#F47118")#, normed=True)
    plt.legend()

    print ('mean values -> high: '+ str(np.std(value_high)) +' ,  low:'+ str(np.mean(value_low)))
    print ('stdev values -> high: '+ str(np.std(value_high)) +' ,  low:'+ str(np.std(value_low)))

    # extract parameters for the distribution corresponding to high and low values.
    lvalParam = [np.mean(value_low), np.std(value_low)]
    hvalParam = [np.mean(value_high), np.std(value_high)]
    print ('low values distribution: mean:' +str(lvalParam[0]) + ', var:' +str(lvalParam[1]) )
    print ('high values distribution: mean:' +str(hvalParam[0]) + ', var:' +str(hvalParam[1]) )
    
    return lvalParam, hvalParam


def ttestsPlot(data1, data2,c1 ='#4F6A9A',c2 = '#AC5255',lab1 = "data1", lab2 = "data2",title = ''):
        # t-TEST
        diff = np.mean(data1) - np.mean(data2)
        [s, p] = stats.ttest_rel(data1,data2)
        print ("Compare: Mean1 = "+ str(np.mean(data1))+ "; Mean2 = "+ str(np.mean(data2))+"; [mean1 - mean2] =  " + str(diff) +"; t =  " + str(round(s,2)) + " ; p-value =" + str(p) )
        
            
        # Set seaborn style for the plot
        fig = plt.figure(figsize=[6,10])
        sns.set(style='white',font_scale=1.5)
        jittr = np.random.uniform(low=-0.3,high=0.3,size=len(data1))    
        plt.scatter([1]*len(data1)+jittr, data1, c= c1, alpha=0.7,label=lab1)
        plt.scatter([2]*len(data2)+jittr, data2, c= c2, alpha=0.7,label=lab2)
        
        ## add lines between slope points in like and dislike for each participant
        
        for i in range(len(data1)):
            plt.plot( [1 + jittr[i],2 + jittr[i]], [ data1[i] , data2[i]],'--', lw=1.0, color = 'black', alpha = 0.2)
            #if data1[i]<data2[i]:
            #    print ("Participants with data1 < data2: " + str(i))
                
        
        #legend(loc = 'best')
        plt.xticks([1, 2,], [lab1, lab2],fontsize=25)
        plt.yticks(fontsize=20)

        plt.ylabel(title, fontsize=28)
        sns.despine()



def plotSigmaBelDensity(trace2, colorP = '#92BDA3', figName = 'figures/sigmaBelDensity_1.svg', n_bins = 20 ):

    # from matplotlib import rcParams
    # rcParams['font.family'] = 'serif'
    # rc("font", family="serif", size=12)
    # rc("text", usetex=True)
    # ppcSigmaBel = pm.sample_posterior_predictive(burned_trace01, samples= samples, model=model_peb01,var_names = ['sigmaBel1','sigmaBel2']) # generates samples from the posterior, im this case the 
    sBel1 = trace2.posterior['sigmaBel1'].values.flatten()
    sBel2 = trace2.posterior['sigmaBel2'].values.flatten()

    ppcSigmaBe1 = sBel1
    ppcSigmaBe2 = sBel2

    #fig, axs = plt.subplots(1, 1, tight_layout=True)
    plt.hist(ppcSigmaBe1, bins=n_bins)
    plt.hist(ppcSigmaBe2, bins=n_bins)
    plt.xlabel("$\sigma_{Bel}$",fontsize=30)
    plt.ylabel("Density",fontsize=30)
    sns.despine()
    
    plt.show() 
    
    # plot Delta SigmaBel distribution
    fig, ax = plt.subplots(figsize=(5,5))
    var1 = ppcSigmaBe2- ppcSigmaBe1
    plt.hist(var1, bins=n_bins, color = '#92BDA3',density=True )
    plt.axvline(0, color='black', lw=3, alpha=0.5,linestyle="--")
    plt.xlabel("$\sigma_{Bel,high}$ - $\sigma_{Bel,low}$",fontsize=30)
    
    kde = stats.gaussian_kde(var1)
    xVar1 = np.linspace(np.min(var1)-0.5, np.max(var1)+0.5, 1000)
    plt.plot(xVar1, kde(xVar1), color = '#080808', lw =1)
    
    plt.ylabel("Density",fontsize=30)
    sns.despine()
    
    # plt.xlim(-np.abs(np.max(var1))- np.abs(np.max(var1))/3, np.abs(np.max(var1))+np.abs(np.max(var1))/3)
    
    #if (np.min(var1) < 0): 
    #    plt.xlim(np.min(var1)-0.1, 0.2)
    #elif (np.min(var1) > 0): 
    #    plt.xlim(-0.2,np.max(var1)+0.1)
    #
    #proportion of the density function above/below zero
    print('Proportion density above 0 : ' + str(np.sum(var1>0)/len(ppcSigmaBe1)))
    print('Proportion density below 0 : ' + str(np.sum(var1<0)/len(ppcSigmaBe1)))
    
       # summary of these distributions
    print ('Mean sigmaBel1:' + str(np.mean(ppcSigmaBe1)) +'; SD sigmaBel1:' + str(np.std(ppcSigmaBe1)) )
    print ('Mean sigmaBel2:' + str(np.mean(ppcSigmaBe2)) +'; SD sigmaBel2:' + str(np.std(ppcSigmaBe2))) 
    print ('Delta sigmaBel mean:' + str(np.mean(ppcSigmaBe2 - ppcSigmaBe1)) + '; SD:' + str(np.std(ppcSigmaBe2 - ppcSigmaBe1)) )

    df_save = pd.DataFrame()
    df_save['sigmaBel1'] =  [ np.mean(ppcSigmaBe1) , np.std(ppcSigmaBe1) ]
    df_save['sigmaBel2'] =  [ np.mean(ppcSigmaBe2) , np.std(ppcSigmaBe2) ]
    df_save['DeltaSigmaBel'] =  [  np.mean(ppcSigmaBe2 - ppcSigmaBe1) , np.std(ppcSigmaBe2 - ppcSigmaBe1)]
    
    
    # df_save.to_csv(figName[:-4] + '_dist_summary.csv')
    
    # plt.savefig(figName, dpi=150)
    
    plt.show()
    