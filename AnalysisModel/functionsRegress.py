import pymc as pm
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
import functionsSims as fs


def exp_genRegression(traceSims,data_part_all_test,rtByPart):

    ####################################
    ## Load simulations ###
    ####################################
    posterior_df = pd.DataFrame()

    # selIt = -1#random.randint(0, ppc.posterior_predictive['choice'].shape[0])
    ppcChoice = traceSims.prior['choice'].values[0].flatten()#[selIt ]
    ppcConf = traceSims.prior['confReport'].values[0].flatten()#[selIt]
    ppcLVal = traceSims.prior['lVal'].values[0].flatten()#[selIt]
    ppcRVal = traceSims.prior['rVal'].values[0].flatten()#[selIt]

    posterior_df['zRT'] = np.tile(rtByPart, len(traceSims.prior['choice'].values[0]) )  # input rt info (from the training set)

    #    dimsD = ppc.predictions['choice'].shape[0] * ppc.predictions['choice'].shape[1]
    posterior_df['choice'] = ppcChoice.flatten()
    posterior_df['conf'] = ppcConf.flatten()
    posterior_df['lValue'] = ppcLVal#.flatten()
    posterior_df['rValue'] = ppcRVal#.flatten()
    posterior_df['dValue'] = posterior_df['rValue'] - posterior_df['lValue']
    posterior_df['sumValue'] = posterior_df['rValue'] + posterior_df['lValue']
    posterior_df['absDValue'] = np.abs(posterior_df['rValue'] - posterior_df['lValue'])
    RbiggerL =posterior_df['rValue'] > posterior_df['lValue']
    posterior_df['correct'] = RbiggerL == posterior_df['choice']

    posterior_df['choValue'] = posterior_df.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    posterior_df['unchoValue'] = posterior_df.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)

    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')  

    # z-score variables to regress
    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')
    posterior_df['zConf'] = fs.zscore1(posterior_df,'conf')
    ####################################
    ## Load participants test trials ###
    ####################################

    data_part = pd.DataFrame()    
    data_part['choice'] = data_part_all_test.ChosenITM
    data_part['dValue'] = data_part_all_test.zRValue - data_part_all_test.zLValue 
    data_part['absDValue'] = np.abs(data_part['dValue'])
    data_part['sumValue'] = data_part_all_test.zRValue + data_part_all_test.zLValue 
    # z-score participant
    data_part['zConf'] = data_part_all_test.zConf.values
    #data_part['zAbsDValue'] =  fs.zscore1(data_part,'absDValue')
    data_part['zAbsDValue'] =  data_part_all_test.zAbsDV.values
    #data_part['zSumValue'] = fs.zscore1(data_part,'sumValue')
    data_part['zSumValue'] = data_part_all_test.zTotVal.values
    # add chosen and unchosen
    data_part['rValue'] = data_part_all_test.zRValue 
    data_part['lValue'] = data_part_all_test.zLValue
    data_part['Part'] = data_part_all_test.Part
    data_part['choValue'] = data_part.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    data_part['unchoValue'] = data_part.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)
    data_part['zChoValue'] = fs.zscore1(data_part,'choValue')
    data_part['zUnchoValue'] = fs.zscore1(data_part,'unchoValue')
    data_part['zRT'] = data_part_all_test.zRT.values



    ##############################  
    ### Generate Regression Plots  
    #################################

    # ## FIGURE
    # ## run simple linear regression for confidence directly in python - - SUM VALUE
    # width_bars = 0.2
    # # get coefficients simulations
    # ols_sims = sm.ols(formula="zConf ~  zAbsDValue + zRT + zSumValue", data=posterior_df).fit()
    # print ('------------SIMS SUMMARY CONF vs SUMVAL +... ------------------ ')
    # print(ols_sims.summary())
    # regParamS = ols_sims.params.values[1:] # Exclude intercept
    # lowLimS = ols_sims.conf_int(alpha=0.05, cols=None)[0].values[1:]
    # highLimS = ols_sims.conf_int(alpha=0.05, cols=None)[1].values[1:]

    # ### get coefficients human
    # ols_human = sm.ols(formula="zConf ~  zAbsDValue + zRT +   zSumValue", data=data_part).fit()
    # print ('------------HUMANS SUMMARY CONF vs SUMVAL +... ------------------ ')
    # print(ols_human.summary())
    # regParamH = ols_human.params.values[1:]
    # lowLimH = ols_human.conf_int(alpha=0.05, cols=None)[0].values[1:]
    # highLimH =ols_human.conf_int(alpha=0.05, cols=None)[1].values[1:]
    # # Add bar plots
    # plt.bar(np.add(range(len(regParamH)),-0.1), regParamH, yerr=np.abs(lowLimH - highLimH)/2,  color=[colorP[0]],width = width_bars,hatch='', label = 'Human')
    # plt.bar(np.add(range(len(regParamS)),0.1), regParamS,yerr=np.abs(lowLimS - highLimS)/2, color=[colorP[1]],width = width_bars,hatch='', label = 'Model Sim')
    # ##
    # #plt.errorbar(range(len(regParamS)),regParamS, yerr= (lowLimS - highLimS)/2 , color='blue',marker='o',fmt='.', markersize=8 , label = 'Sims');
    # #plt.errorbar(range(len(regParamH)),regParamH, yerr= (lowLimH - highLimH)/2 , color='red',marker='o',fmt='.', markersize=8 , label = 'Human');
    # plt.axhline(0, color='black', lw=2, alpha=0.5, linestyle = 'dotted')

    # plt.xlabel("Parameters")
    # plt.ylabel("Regression Coefficient")
    # plt.title('Confidence Simulation')
    # plt.xticks([0,1,2],[r'$|\Delta$Value$|$','RT','$\Sigma$Value'])
    # plt.legend(frameon=False)
    # sns.despine()

    # plt.show()

    ## FIGURE for Chosen UnChosen effect on confidence ----------------
    ## run simple linear regression for confidence directly in python
    posterior_df['zChoValue'] = fs.zscore1(posterior_df,'choValue')
    posterior_df['zUnchoValue'] = fs.zscore1(posterior_df,'unchoValue')

    # get coefficients simulations
    print ('------------SIMS SUMMARY CONF vs chosen + unchosen ------------------ ')
    ols_sims_overall = sm.ols(formula="zConf ~ zRT +  zChoValue + zUnchoValue", data=posterior_df).fit()
    print(ols_sims_overall.summary())

    regParamSov = ols_sims_overall.params.values[1:]
    lowLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[1].values[1:]

    ### get coefficients human
    print ('------------HUMANS SUMMARY CONF vs chosen + unchosen ------------------ ')
    ols_human = sm.ols(formula="zConf ~  zRT + zChoValue + zUnchoValue", data=data_part).fit()
    print(ols_human.summary())
    regParamH = ols_human.params.values[1:]
    lowLimH = ols_human.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimH = ols_human.conf_int(alpha=0.05, cols=None)[1].values[1:]
    ##
    # plt.bar(np.add(range(len(regParamH)),-0.1), regParamH,yerr=np.abs(lowLimH - highLimH)/2, color=[colorP[0]],width = width_bars,hatch='', label = 'Human')
    # plt.bar(np.add(range(len(regParamSov)),0.1), regParamSov, yerr=np.abs(lowLimSov - highLimSov)/2,  color=[colorP[1]],width = width_bars,hatch='', label = 'Model Sim')
    # plt.axhline(0, color='black', lw=2, alpha=0.5)

    # plt.xlabel("Parameters")
    # plt.ylabel("Regression Coefficient")
    # plt.title('Confidence Simulation')
    # plt.xticks([0,1,2],['RT','ChoValue','UnchoValue'])
    # plt.legend(frameon=False)
    # sns.despine()

    # plt.show()      

    return regParamH, lowLimH, highLimH, regParamSov, lowLimSov, highLimSov



def exp_genRegression_withDifficulty(traceSims,data_part_all_test,rtByPart):

    ####################################
    ## Load simulations ###
    ####################################
    posterior_df = pd.DataFrame()

    # selIt = -1#random.randint(0, ppc.posterior_predictive['choice'].shape[0])
    ppcChoice = traceSims.prior['choice'].values[0].flatten()#[selIt ]
    ppcConf = traceSims.prior['confReport'].values[0].flatten()#[selIt]
    ppcLVal = traceSims.prior['lVal'].values[0].flatten()#[selIt]
    ppcRVal = traceSims.prior['rVal'].values[0].flatten()#[selIt]

    posterior_df['zRT'] = np.tile(rtByPart, len(traceSims.prior['choice'].values[0]) )  # input rt info (from the training set)

    #    dimsD = ppc.predictions['choice'].shape[0] * ppc.predictions['choice'].shape[1]
    posterior_df['choice'] = ppcChoice.flatten()
    posterior_df['conf'] = ppcConf.flatten()
    posterior_df['lValue'] = ppcLVal#.flatten()
    posterior_df['rValue'] = ppcRVal#.flatten()
    posterior_df['dValue'] = posterior_df['rValue'] - posterior_df['lValue']
    posterior_df['sumValue'] = posterior_df['rValue'] + posterior_df['lValue']
    posterior_df['absDValue'] = np.abs( posterior_df['dValue'])
    RbiggerL =posterior_df['rValue'] > posterior_df['lValue']
    posterior_df['correct'] = RbiggerL == posterior_df['choice']

    posterior_df['choValue'] = posterior_df.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    posterior_df['unchoValue'] = posterior_df.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)

    # posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zAbsDValue'] = posterior_df['absDValue']

    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')  

    # z-score variables to regress
    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')
    posterior_df['zConf'] = fs.zscore1(posterior_df,'conf')
    ####################################
    ## Load participants test trials ###
    ####################################

    data_part = pd.DataFrame()    
    data_part['choice'] = data_part_all_test.ChosenITM
    data_part['dValue'] = data_part_all_test.zRValue - data_part_all_test.zLValue 
    data_part['absDValue'] = np.abs(data_part['dValue'])
    data_part['sumValue'] = data_part_all_test.zRValue + data_part_all_test.zLValue 
    # z-score participant
    data_part['zConf'] = data_part_all_test.zConf.values
    data_part['zAbsDValue'] =  fs.zscore1(data_part,'absDValue')
    # data_part['zAbsDValue'] =  data_part_all_test.zAbsDV.values
    #data_part['zSumValue'] = fs.zscore1(data_part,'sumValue')
    data_part['zSumValue'] = data_part_all_test.zTotVal.values
    # add chosen and unchosen
    data_part['rValue'] = data_part_all_test.zRValue 
    data_part['lValue'] = data_part_all_test.zLValue
    data_part['Part'] = data_part_all_test.Part
    data_part['choValue'] = data_part.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    data_part['unchoValue'] = data_part.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)
    data_part['zChoValue'] = fs.zscore1(data_part,'choValue')
    data_part['zUnchoValue'] = fs.zscore1(data_part,'unchoValue')
    data_part['zRT'] = data_part_all_test.zRT.values



    ##############################  
    ### Generate Regression Plots  
    #################################


    ## FIGURE for Chosen UnChosen effect on confidence ----------------
    ## run simple linear regression for confidence directly in python
    posterior_df['zChoValue'] = fs.zscore1(posterior_df,'choValue')
    posterior_df['zUnchoValue'] = fs.zscore1(posterior_df,'unchoValue')

    # get coefficients simulations
    print ('------------SIMS SUMMARY CONF vs chosen + unchosen ------------------ ')
    ols_sims_overall = sm.ols(formula="zConf ~ zAbsDValue + zRT +  zChoValue + zUnchoValue ", data=posterior_df).fit()
    print(ols_sims_overall.summary())

    regParamSov = ols_sims_overall.params.values[1:]
    lowLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[1].values[1:]

    ### get coefficients human
    print ('------------HUMANS SUMMARY CONF vs chosen + unchosen ------------------ ')
    ols_human = sm.ols(formula="zConf ~ zAbsDValue +  zRT + zChoValue + zUnchoValue", data=data_part).fit()
    print(ols_human.summary())
    regParamH = ols_human.params.values[1:]
    lowLimH = ols_human.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimH = ols_human.conf_int(alpha=0.05, cols=None)[1].values[1:]
    ##
    # plt.bar(np.add(range(len(regParamH)),-0.1), regParamH,yerr=np.abs(lowLimH - highLimH)/2, color=[colorP[0]],width = width_bars,hatch='', label = 'Human')
    # plt.bar(np.add(range(len(regParamSov)),0.1), regParamSov, yerr=np.abs(lowLimSov - highLimSov)/2,  color=[colorP[1]],width = width_bars,hatch='', label = 'Model Sim')
    # plt.axhline(0, color='black', lw=2, alpha=0.5)

    # plt.xlabel("Parameters")
    # plt.ylabel("Regression Coefficient")
    # plt.title('Confidence Simulation')
    # plt.xticks([0,1,2],['RT','ChoValue','UnchoValue'])
    # plt.legend(frameon=False)
    # sns.despine()

    # plt.show()      

    return regParamH, lowLimH, highLimH, regParamSov, lowLimSov, highLimSov



def exp_genRegression_withSum(traceSims,data_part_all_test,rtByPart):

    ####################################
    ## Load simulations ###
    ####################################
    posterior_df = pd.DataFrame()

    # selIt = -1#random.randint(0, ppc.posterior_predictive['choice'].shape[0])
    ppcChoice = traceSims.prior['choice'].values[0].flatten()#[selIt ]
    ppcConf = traceSims.prior['confReport'].values[0].flatten()#[selIt]
    ppcLVal = traceSims.prior['lVal'].values[0].flatten()#[selIt]
    ppcRVal = traceSims.prior['rVal'].values[0].flatten()#[selIt]

    posterior_df['zRT'] = np.tile(rtByPart, len(traceSims.prior['choice'].values[0]) )  # input rt info (from the training set)

    #    dimsD = ppc.predictions['choice'].shape[0] * ppc.predictions['choice'].shape[1]
    posterior_df['choice'] = ppcChoice.flatten()
    posterior_df['conf'] = ppcConf.flatten()
    posterior_df['lValue'] = ppcLVal#.flatten()
    posterior_df['rValue'] = ppcRVal#.flatten()
    posterior_df['dValue'] = posterior_df['rValue'] - posterior_df['lValue']
    posterior_df['sumValue'] = posterior_df['rValue'] + posterior_df['lValue']
    posterior_df['absDValue'] = np.abs( posterior_df['dValue'])
    RbiggerL =posterior_df['rValue'] > posterior_df['lValue']
    posterior_df['correct'] = RbiggerL == posterior_df['choice']

    posterior_df['choValue'] = posterior_df.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    posterior_df['unchoValue'] = posterior_df.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)

    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')  

    # z-score variables to regress
    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')
    posterior_df['zConf'] = fs.zscore1(posterior_df,'conf')
    ####################################
    ## Load participants test trials ###
    ####################################

    data_part = pd.DataFrame()    
    data_part['choice'] = data_part_all_test.ChosenITM
    data_part['dValue'] = data_part_all_test.zRValue - data_part_all_test.zLValue 
    data_part['absDValue'] = np.abs(data_part['dValue'])
    data_part['sumValue'] = data_part_all_test.zRValue + data_part_all_test.zLValue 
    # z-score participant
    data_part['zConf'] = data_part_all_test.zConf.values
    data_part['zAbsDValue'] =  fs.zscore1(data_part,'absDValue')
    # data_part['zAbsDValue'] =  data_part_all_test.zAbsDV.values
    #data_part['zSumValue'] = fs.zscore1(data_part,'sumValue')
    data_part['zSumValue'] = data_part_all_test.zTotVal.values
    # add chosen and unchosen
    data_part['rValue'] = data_part_all_test.zRValue 
    data_part['lValue'] = data_part_all_test.zLValue
    data_part['Part'] = data_part_all_test.Part
    data_part['choValue'] = data_part.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    data_part['unchoValue'] = data_part.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)
    data_part['zChoValue'] = fs.zscore1(data_part,'choValue')
    data_part['zUnchoValue'] = fs.zscore1(data_part,'unchoValue')
    data_part['zRT'] = data_part_all_test.zRT.values



    ##############################  
    ### Generate Regression Plots  
    #################################

    ## FIGURE for Chosen UnChosen effect on confidence ----------------
    ## run simple linear regression for confidence directly in python
    posterior_df['zChoValue'] = fs.zscore1(posterior_df,'choValue')
    posterior_df['zUnchoValue'] = fs.zscore1(posterior_df,'unchoValue')

    # get coefficients simulations
    print ('------------SIMS SUMMARY CONF vs DEvSum ------------------ ')
    ols_sims_overall = sm.ols(formula="zConf ~ zAbsDValue + zRT + zSumValue", data=posterior_df).fit()
    print(ols_sims_overall.summary())

    regParamSov = ols_sims_overall.params.values[1:]
    lowLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[1].values[1:]

    ### get coefficients human
    print ('------------HUMANS SUMMARY CONF vs DEvSum ------------------ ')
    ols_human = sm.ols(formula="zConf ~ zAbsDValue + zRT + zSumValue", data=data_part).fit()
    print(ols_human.summary())
    regParamH = ols_human.params.values[1:]
    lowLimH = ols_human.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimH = ols_human.conf_int(alpha=0.05, cols=None)[1].values[1:]
    ##
    # plt.bar(np.add(range(len(regParamH)),-0.1), regParamH,yerr=np.abs(lowLimH - highLimH)/2, color=[colorP[0]],width = width_bars,hatch='', label = 'Human')
    # plt.bar(np.add(range(len(regParamSov)),0.1), regParamSov, yerr=np.abs(lowLimSov - highLimSov)/2,  color=[colorP[1]],width = width_bars,hatch='', label = 'Model Sim')
    # plt.axhline(0, color='black', lw=2, alpha=0.5)

    # plt.xlabel("Parameters")
    # plt.ylabel("Regression Coefficient")
    # plt.title('Confidence Simulation')
    # plt.xticks([0,1,2],['RT','ChoValue','UnchoValue'])
    # plt.legend(frameon=False)
    # sns.despine()

    # plt.show()      

    return regParamH, lowLimH, highLimH, regParamSov, lowLimSov, highLimSov




def exp_genRegression_withSumCoh(traceSims,data_part_all_test,rtByPart,cohByPart):

    ####################################
    ## Load simulations ###
    ####################################
    posterior_df = pd.DataFrame()

    # selIt = -1#random.randint(0, ppc.posterior_predictive['choice'].shape[0])
    ppcChoice = traceSims.prior['choice'].values[0].flatten()#[selIt ]
    ppcConf = traceSims.prior['confReport'].values[0].flatten()#[selIt]
    ppcLVal = traceSims.prior['lVal'].values[0].flatten()#[selIt]
    ppcRVal = traceSims.prior['rVal'].values[0].flatten()#[selIt]

    posterior_df['zRT'] = np.tile(rtByPart, len(traceSims.prior['choice'].values[0]) )  # input rt info (from the training set)
    posterior_df['zCoherence'] = np.tile(cohByPart, len(traceSims.prior['choice'].values[0]) )  # input rt info (from the training set)

    #    dimsD = ppc.predictions['choice'].shape[0] * ppc.predictions['choice'].shape[1]
    posterior_df['choice'] = ppcChoice.flatten()
    posterior_df['conf'] = ppcConf.flatten()
    posterior_df['lValue'] = ppcLVal#.flatten()
    posterior_df['rValue'] = ppcRVal#.flatten()
    posterior_df['dValue'] = posterior_df['rValue'] - posterior_df['lValue']
    posterior_df['sumValue'] = posterior_df['rValue'] + posterior_df['lValue']
    posterior_df['absDValue'] = np.abs( posterior_df['dValue'])
    RbiggerL =posterior_df['rValue'] > posterior_df['lValue']
    posterior_df['correct'] = RbiggerL == posterior_df['choice']

    posterior_df['choValue'] = posterior_df.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    posterior_df['unchoValue'] = posterior_df.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)

    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')  

    # z-score variables to regress
    posterior_df['zAbsDValue'] = fs.zscore1(posterior_df,'absDValue')
    posterior_df['zSumValue'] = fs.zscore1(posterior_df,'sumValue')
    posterior_df['zConf'] = fs.zscore1(posterior_df,'conf')
    ####################################
    ## Load participants test trials ###
    ####################################

    data_part = pd.DataFrame()    
    data_part['choice'] = data_part_all_test.ChosenITM
    data_part['dValue'] = data_part_all_test.zRValue - data_part_all_test.zLValue 
    data_part['absDValue'] = np.abs(data_part['dValue'])
    data_part['sumValue'] = data_part_all_test.zRValue + data_part_all_test.zLValue 
    # z-score participant
    data_part['zConf'] = data_part_all_test.zConf.values
    data_part['zAbsDValue'] =  fs.zscore1(data_part,'absDValue')
    # data_part['zAbsDValue'] =  data_part_all_test.zAbsDV.values
    #data_part['zSumValue'] = fs.zscore1(data_part,'sumValue')
    data_part['zSumValue'] = data_part_all_test.zTotVal.values
    # add chosen and unchosen
    data_part['rValue'] = data_part_all_test.zRValue 
    data_part['lValue'] = data_part_all_test.zLValue
    data_part['Part'] = data_part_all_test.Part
    data_part['choValue'] = data_part.apply(lambda row: row.rValue if row.choice==1 else row.lValue, axis=1)
    data_part['unchoValue'] = data_part.apply(lambda row: row.lValue if row.choice==1 else row.rValue, axis=1)
    data_part['zChoValue'] = fs.zscore1(data_part,'choValue')
    data_part['zUnchoValue'] = fs.zscore1(data_part,'unchoValue')
    data_part['zRT'] = data_part_all_test.zRT.values
    data_part['zCoherence'] = data_part_all_test.zCoherence.values



    ##############################  
    ### Generate Regression Plots  
    #################################

    ## FIGURE for Chosen UnChosen effect on confidence ----------------
    ## run simple linear regression for confidence directly in python
    posterior_df['zChoValue'] = fs.zscore1(posterior_df,'choValue')
    posterior_df['zUnchoValue'] = fs.zscore1(posterior_df,'unchoValue')

    # get coefficients simulations
    print ('------------SIMS SUMMARY CONF vs DEvSum ------------------ ')
    ols_sims_overall = sm.ols(formula="zConf ~ zCoherence + zAbsDValue + zRT + zSumValue", data=posterior_df).fit()
    print(ols_sims_overall.summary())

    regParamSov = ols_sims_overall.params.values[1:]
    lowLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimSov = ols_sims_overall.conf_int(alpha=0.05, cols=None)[1].values[1:]

    ### get coefficients human
    print ('------------HUMANS SUMMARY CONF vs DEvSum ------------------ ')
    ols_human = sm.ols(formula="zConf ~ zCoherence + zAbsDValue + zRT + zSumValue", data=data_part).fit()
    print(ols_human.summary())
    regParamH = ols_human.params.values[1:]
    lowLimH = ols_human.conf_int(alpha=0.05, cols=None)[0].values[1:]
    highLimH = ols_human.conf_int(alpha=0.05, cols=None)[1].values[1:]
    ##
    # plt.bar(np.add(range(len(regParamH)),-0.1), regParamH,yerr=np.abs(lowLimH - highLimH)/2, color=[colorP[0]],width = width_bars,hatch='', label = 'Human')
    # plt.bar(np.add(range(len(regParamSov)),0.1), regParamSov, yerr=np.abs(lowLimSov - highLimSov)/2,  color=[colorP[1]],width = width_bars,hatch='', label = 'Model Sim')
    # plt.axhline(0, color='black', lw=2, alpha=0.5)

    # plt.xlabel("Parameters")
    # plt.ylabel("Regression Coefficient")
    # plt.title('Confidence Simulation')
    # plt.xticks([0,1,2],['RT','ChoValue','UnchoValue'])
    # plt.legend(frameon=False)
    # sns.despine()

    # plt.show()      

    return regParamH, lowLimH, highLimH, regParamSov, lowLimSov, highLimSov
