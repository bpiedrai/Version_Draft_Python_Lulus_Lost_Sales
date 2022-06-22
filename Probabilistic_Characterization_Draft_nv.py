#------------------------------------------------------------------------------------------------------------------------
#
# MODULE FOR THE PROBABILISTIC CHARACTERIZATION OF THE STATISTICAL DATA OF SALES FOR A SKU (type/size)
#
# Characterization identifies inner probabilistic seasons inside an sales season, calculating the expected value of daily
# sales for each inner season.
#
#------------------------------------------------------------------------------------------------------------------------

# Import of the required libraries

import csv
import math
import pandas as pd
from pydoc import describe
from pandas import Series, DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import inspect
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import snowflake
from scipy import stats
from scipy.stats import betaprime
from scipy.stats import burr
from scipy.stats import burr12
from scipy.stats import chi
from scipy.stats import chi2
from scipy.stats import erlang
from scipy.stats import expon
from scipy.stats import exponpow
from scipy.stats import exponweib
from scipy.stats import fatiguelife
from scipy.stats import fisk
from scipy.stats import foldnorm
from scipy.stats import gamma
from scipy.stats import genexpon
from scipy.stats import gengamma
from scipy.stats import genhalflogistic
from scipy.stats import geninvgauss
from scipy.stats import genpareto
from scipy.stats import gilbrat
from scipy.stats import gompertz
from scipy.stats import halfgennorm
from scipy.stats import halflogistic
from scipy.stats import halfnorm
from scipy.stats import invgamma
from scipy.stats import invgauss
from scipy.stats import invweibull
from scipy.stats import kappa3
from scipy.stats import kstwobign
from scipy.stats import loglaplace
from scipy.stats import lognorm
from scipy.stats import lomax
from scipy.stats import maxwell
from scipy.stats import mielke
from scipy.stats import nakagami
from scipy.stats import ncf
from scipy.stats import ncx2
from scipy.stats import powerlognorm
from scipy.stats import rayleigh
from scipy.stats import recipinvgauss
from scipy.stats import rice
from scipy.stats import truncexpon
from scipy.stats import wald
from scipy.stats import weibull_min

#
# FUNCTION TO REMOVE OUTLIERS FROM A DATA SET
#
# Input Data:   data_in ------> data set from which outliers will be removed
#               set_data_in --> complete data set, which includes a column with data_in and all additional information columns
#
# Ouput Data:   max_val_ref -----------------> the maximum value in data set after removing outliers
#               complete_set_data_in_woout --> it comes from set_data_in but removing all those rows corresponding to outliers
#
# Method used for outliers removal is MEDA criteria, which states that a value is considered as an outlier if:
#
#                           abs(Xi - med(X))/MEDA > 4.5
#
# where:    Xi is a value in data_in
#           X is data_in
#           med(X) is the median of data_in
#           abs(Xi - med(X)) is the absolut deviation of a value in data_in
#           MEDA is the median of the absolute deviations of X
#

def out_rem(data_in, complete_set_data_in):
    data_in_ord = data_in.sort_values().reset_index(drop=True)
    data_in_ord_med = data_in_ord.median()
    abs_des_dio = [abs(x - data_in_ord_med) for x in data_in_ord]
    abs_des_dio_df = pd.Series(abs_des_dio)
    MEDA_abs_des_dio_df = abs_des_dio_df.median()
    if MEDA_abs_des_dio_df != 0:
        out_cri = [(x/MEDA_abs_des_dio_df) for x in abs_des_dio_df]
    else:
        out_cri = [(x / 2) for x in abs_des_dio_df]
    out_cri_df = pd.DataFrame(out_cri)
    out_cri_idx=out_cri_df[out_cri_df[0]>4.5].index
    if out_cri_idx.size == 0:
        max_val_ref = data_in.max()
        complete_set_data_in_woout = complete_set_data_in
    else:
        min_out = data_in_ord[out_cri_idx[0]]
        out_idx=complete_set_data_in[complete_set_data_in[1]>min_out].index
        complete_set_data_in_woout = complete_set_data_in.drop(out_idx).reset_index(drop=True)
        max_val_ref = complete_set_data_in_woout[1].max()
    return max_val_ref, complete_set_data_in_woout

#
# FUNCTION TO SELECT A PROBABILITY DISTRIBUTION FUNCTION AMONG ALL DISTRIBUTIONS AVAILABLE IN LIBRARY STATS FROM SCIPY
#
# Input Data:   family='realplus' ------>   realplus corresponds to the probability distribution functions with zero or positive arguments,
#                                           which corresponds to the ones we are interested in because demand of items is not negative
#
# Ouput Data:   selection -----------------> List of main distributions in 'realplus', including their domain
#

def distribution_selection(family='realplus', verbose=True):
    main_distributions = [getattr(stats, di) for di in dir(stats) \
                          if isinstance(getattr(stats, di), (stats.rv_continuous, stats.rv_discrete))]

    exclusions = ['geninvgauss', 'alpha', 'studentized_range', 'vonmises', 'levy_stable', 'erlang', 'gamma', 'rayleigh']
    main_distributions = [dist for dist in main_distributions if dist.name not in exclusions]

    domain = {
        'realall': [-np.inf, np.inf],
        'realline': [np.inf, np.inf],
        'realplus': [0, np.inf],
        'real0to1': [0, 1],
        'discrete': [None, None], }

    distribution = []
    type = []
    domain_lower = []
    domain_upper = []

    for dist in main_distributions:
        distribution.append(dist.name)
        type.append(np.where(isinstance(dist, stats.rv_continuous), 'continuos', 'discrete'))
        domain_lower.append(dist.a)
        domain_upper.append(dist.b)

    info_main_distributions = pd.DataFrame({
        'distribution': distribution,
        'type': type,
        'domain_lower': domain_lower,
        'domain_upper': domain_upper
    })

    info_main_distributions = info_main_distributions \
        .sort_values(by=['domain_lower', 'domain_upper']) \
        .reset_index(drop=True)

    if family in ['realplus']:
        info_main_distributions = info_main_distributions[info_main_distributions['type'] == 'continuos']
        condition = (info_main_distributions['domain_lower'] == domain[family][0]) & \
                    (info_main_distributions['domain_upper'] == domain[family][1])
        info_main_distributions = info_main_distributions[condition].reset_index(drop=True)

    if family in ['discrete']:
        info_main_distributions = info_main_distributions[info_main_distributions['type'] == 'discrete']

    selection = [dist for dist in main_distributions \
                 if dist.name in info_main_distributions['distribution'].values]


    return selection

#
# FUNCTION TO COMPARE DISTRIBUTIONS AFTER FITTING TO SELECT THE BEST ONE
#
# Input Data:   x --->---> corresponds to the data set which is going to be fitted to a probability distribution function
#               family --> corresponds to the set of probability distribution functions to be fitted to the input data and
#                          compared to select the best one
#               orden ---> corresponds to the method to be used for ordering the fitted distributions and select the best one.
#                          Akaike Generalized Information Criterion (aic) is used.
#
# Ouput Data:   results -> gives the best fitted probability distribution function, according to aic.  It also gives
#               the parameters of the probability distribution function and its expected value.
#
# Method used for selecting the best probability distribution fitting is known as Maximum Likelihood Estimation, using the
# already mentioned Akaike Generalized Information Criterion, which is given by:
#
#                           GAIC = -2*likelihood * 2*number of parameters
#
# as more negative is GAIC, better is the distribution fitting.
# (Bayesian Information Criterion -bic- coul be also used)
#

def comparing_distributions(x: object, family: object = 'realplus', orden: object = 'aic',
                            verbose: object = True) -> object:
    distributions = distribution_selection(family=family, verbose=verbose)
    distribution_ = []
    log_likelihood_ = []
    aic_ = []
    bic_ = []
    exp_value = []
    n_parameters_ = []
    parameters_ = []

    for i, distribution in enumerate(distributions):

# The following code line should be implemented in case there is a need to look at the probability distribution functions that is
# being fitted.
#        print(f"{i + 1}/{len(distributions)} Fitting distribution: {distribution.name}")
#

        try:
            parameters = distribution.fit(data=x)
            name_parameters = [p for p in inspect.signature(distribution._pdf).parameters \
                               if not p == 'x'] + ["loc", "scale"]
            parameters_dict = dict(zip(name_parameters, parameters))
            log_likelihood = distribution.logpdf(x, *parameters).sum()
            aic = -2 * log_likelihood + 2 * len(parameters)
            bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parameters)

            distribution_.append(distribution.name)
            log_likelihood_.append(log_likelihood)
            aic_.append(aic)
            bic_.append(bic)
            n_parameters_.append(len(parameters))
            parameters_.append(parameters_dict)
            if distribution.name == 'betaprime':
                stat_data = betaprime.stats(*parameters, moments='mvsk')
            elif distribution.name == 'burr':
                stat_data = burr.stats(*parameters, moments='mvsk')
            elif distribution.name == 'burr12':
                stat_data = burr12.stats(*parameters, moments='mvsk')
            elif distribution.name == 'chi':
                stat_data = chi.stats(*parameters, moments='mvsk')
            elif distribution.name == 'chi2':
                stat_data = chi2.stats(*parameters, moments='mvsk')
            elif distribution.name == 'erlang':
                stat_data = erlang.stats(*parameters, moments='mvsk')
            elif distribution.name == 'expon':
                stat_data = expon.stats(*parameters, moments='mvsk')
            elif distribution.name == 'exponpow':
                stat_data = exponpow.stats(*parameters, moments='mvsk')
            elif distribution.name == 'exponweib':
                stat_data = exponweib.stats(*parameters, moments='mvsk')
            elif distribution.name == 'fatiguelife':
                stat_data = fatiguelife.stats(*parameters, moments='mvsk')
            elif distribution.name == 'fisk':
                stat_data = fisk.stats(*parameters, moments='mvsk')
            elif distribution.name == 'foldnorm':
                stat_data = foldnorm.stats(*parameters, moments='mvsk')
            elif distribution.name == 'gamma':
                stat_data = gamma.stats(*parameters, moments='mvsk')
            elif distribution.name == 'genexpon':
                stat_data = genexpon.stats(*parameters, moments='mvsk')
            elif distribution.name == 'gengamma':
                stat_data = gengamma.stats(*parameters, moments='mvsk')
            elif distribution.name == 'genhalflogistic':
                stat_data = genhalflogistic.stats(*parameters, moments='mvsk')
            elif distribution.name == 'geninvgauss':
                stat_data = geninvgauss.stats(*parameters, moments='mvsk')
            elif distribution.name == 'genpareto':
                stat_data = genpareto.stats(*parameters, moments='mvsk')
            elif distribution.name == 'gilbrat':
                stat_data = gilbrat.stats(*parameters, moments='mvsk')
            elif distribution.name == 'gompertz':
                stat_data = gompertz.stats(*parameters, moments='mvsk')
            elif distribution.name == 'halfgennorm':
                stat_data = halfgennorm.stats(*parameters, moments='mvsk')
            elif distribution.name == 'halflogistic':
                stat_data = halflogistic.stats(*parameters, moments='mvsk')
            elif distribution.name == 'halfnorm':
                stat_data = halfnorm.stats(*parameters, moments='mvsk')
            elif distribution.name == 'invgamma':
                stat_data = invgamma.stats(*parameters, moments='mvsk')
            elif distribution.name == 'invgauss':
                stat_data = invgauss.stats(*parameters, moments='mvsk')
            elif distribution.name == 'invweibull':
                stat_data = invweibull.stats(*parameters, moments='mvsk')
            elif distribution.name == 'kappa3':
                stat_data = kappa3.stats(*parameters, moments='mvsk')
            elif distribution.name == 'kstwobign':
                stat_data = kstwobign.stats(*parameters, moments='mvsk')
            elif distribution.name == 'loglaplace':
                stat_data = loglaplace.stats(*parameters, moments='mvsk')
            elif distribution.name == 'lognorm':
                stat_data = lognorm.stats(*parameters, moments='mvsk')
            elif distribution.name == 'lomax':
                stat_data = lomax.stats(*parameters, moments='mvsk')
            elif distribution.name == 'maxwell':
                stat_data = maxwell.stats(*parameters, moments='mvsk')
            elif distribution.name == 'mielke':
                stat_data = mielke.stats(*parameters, moments='mvsk')
            elif distribution.name == 'nakagami':
                stat_data = nakagami.stats(*parameters, moments='mvsk')
            elif distribution.name == 'ncf':
                stat_data = ncf.stats(*parameters, moments='mvsk')
            elif distribution.name == 'ncx2':
                stat_data = ncx2.stats(*parameters, moments='mvsk')
            elif distribution.name == 'powerlognorm':
                stat_data = powerlognorm.stats(*parameters, moments='mvsk')
            elif distribution.name == 'rayleigh':
                stat_data = rayleigh.stats(*parameters, moments='mvsk')
            elif distribution.name == 'recipinvgauss':
                stat_data = recipinvgauss.stats(*parameters, moments='mvsk')
            elif distribution.name == 'rice':
                stat_data = rice.stats(*parameters, moments='mvsk')
            elif distribution.name == 'truncexpon':
                stat_data = truncexpon.stats(*parameters, moments='mvsk')
            elif distribution.name == 'wald':
                stat_data = wald.stats(*parameters, moments='mvsk')
            elif distribution.name == 'weibull_min':
                stat_data = weibull_min.stats(*parameters, moments='mvsk')
            stat_data = list(stat_data)
            exp_value_array = stat_data[0] * 1
            exp_value.append(exp_value_array)

            results = pd.DataFrame({
                'distribution': distribution_,
                'log_likelihood': log_likelihood_,
                'aic': aic_,
                'bic': bic_,
                'n_parameters': n_parameters_,
                'parameters': parameters_,
                'exp_value': exp_value
            })

            results = results.sort_values(by=orden).reset_index(drop=True)

        except  Exception as e:
            print(f"Error at fitting distribution {distribution.name}")
            print(e)
            print("")

    return results


#
# FUNCTION TO PLOT THE HISTOGRAM OF FRECUENCIES OF INPUT DATA AND A FITTED PROBABILITY DISTRIBUTION FUNCTION
#
# Input Data:   x --->---> corresponds to the data set from which a histogram of frequencies is going to be plotted
#               distribution_name --> corresponds to the probability distribution function to be plotted along with the histogram
#               of frequencies of input data.
#
# Ouput Data:   Plot of the histogram of frequencies of input data along with the plot of the probability distribution function
#

def plot_distribution(x, distribution_name, ax=None):
    distribution = getattr(stats, distribution_name)

    parameters = distribution.fit(data=x)

    parameters_name = [p for p in inspect.signature(distribution._pdf).parameters \
                       if not p == 'x'] + ["loc", "scale"]
    parameters_dict = dict(zip(parameters_name, parameters))

    log_likelihood = distribution.logpdf(x, *parameters).sum()

    aic = -2 * log_likelihood + 2 * len(parameters)
    bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parameters)

    x_hat = np.linspace(min(x), max(x), num=1000)
    y_hat = distribution.pdf(x_hat, *parameters)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_hat, y_hat, linewidth=2, label=distribution.name)
    ax.hist(x=x, density=True, bins=10, color="#3182bd");
    ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    ax.set_title("Distribution Fit")
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend();

    print('Fitting Results')
    print(f"Distribution:   {distribution.name}")
    print(f"Domain:         {[distribution.a, distribution.b]}")
    print(f"Parameters:     {parameters_dict}")
    print(f"Log likelihood: {log_likelihood}")
    print(f"AIC:            {aic}")
    print(f"BIC:            {bic}")

    plt.savefig('DistFit.png')

    return ax

#
# FUNCTION TO PLOT A SET OF PROBABILITY DISTRIBUTION FUNCTIONS
#
# This function could be used to plot the set of best fitted probability distribution functions
#
# Input Data:   x ---> corresponds to the data set from which the histogram of frquencies is going to be plotted
#               distributions_name --> contains the list of probability distribution functions to be plotted.
#
# Ouput Data:   Plot of the histogram of frequencies of input data along with the plot of the probability distribution
#               functions included in distributions_name
#

def plot_multiple_distributions(x, distributions_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(x=x, density=True, bins=10, color="#3182bd")
    ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    ax.set_title('Distributions Fit')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')

    for name in distributions_name:
        distribution = getattr(stats, name)

        parameters = distribution.fit(data=x)

        name_parameters = [p for p in inspect.signature(distribution._pdf).parameters \
                           if not p == 'x'] + ["loc", "scale"]
        parameters_dict = dict(zip(name_parameters, parameters))

        log_likelihood = distribution.logpdf(x, *parameters).sum()

        aic = -2 * log_likelihood + 2 * len(parameters)
        bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parameters)

        x_hat = np.linspace(min(x), max(x), num=1000)
        y_hat = distribution.pdf(x_hat, *parameters)
        ax.plot(x_hat, y_hat, linewidth=2, label=distribution.name)

    ax.legend();

    plt.savefig('MainDistFit.png')

    return ax


#
# FUNCTION TO PERFORM THE PROBABILISTIC CHARACTERIZATION OF A DATA SET, WHICH MEANS TO FIND THE BEST PROBABILITY DISTRIBUTION
# FUNCTION THAT FITS THE DATA, ITS PARAMETERS AND ITS EXPECTED VALUES
#
# Input Data:   data ---------->    Input data to be characterized
#               graph_req ----->    Parameter to define if a plot of the Empiric Distribution of Demand and the Empirical Distribution
#                                   Function is required for input data.  Should be set to 1 if plot is required, 0 if not.
#               info_req ------>    Parameter to define if a table of basic information of the best fitted Probability Distribution
#                                   Function is required to be printted.  Should be set to 1 if printing is required, 0 if not.
#               graph_req_ad -->    Parameter to define if a plot of the best 5 fitted Probability Distribution Functions is
#                                   required along with the histogram of frquencies of input data.
#
# Output Data:  Probabilitic Characterization of a Data Set, expected value mainly, which along distribution name and
#               distribution parameters is included in results.
#


def prob_char_1sku(data, graph_req, info_req, graph_req_ad):

    warnings.filterwarnings('ignore')

    demand = data

    if info_req == 1:
        main_distributions = [getattr(stats, m_d) for m_d in dir(stats) \
                if isinstance(getattr(stats, m_d), (stats.rv_continuous, stats.rv_discrete))]

        distribution = []
        lower_domain = []
        upper_domain = []

        for di in main_distributions:
            distribution.append(di.name)
            lower_domain.append(di.a)
            upper_domain.append(di.b)

        info_distributions = pd.DataFrame({
            'distribution': distribution,
            'lower_domain': lower_domain,
            'upper_domain': upper_domain})

        info_distributions = info_distributions \
            .sort_values(by=['lower_domain', 'upper_domain']) \
            .reset_index(drop=True)

        print(info_distributions)

    if graph_req == 1:
        plt.rcParams['savefig.bbox'] = "tight"
        style.use('ggplot') or plt.style.use('ggplot')

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        axs[0].hist(x=demand, bins=10, color="#3182bd")
        axs[0].plot(demand, np.full_like(demand, -0.01), '|k', markeredgewidth=1)
        axs[0].set_title('Empiric distribution of Demand')
        axs[0].set_xlabel('Demand')
        axs[0].set_ylabel('Counts')

        ecdf = ECDF(x=demand)
        axs[1].plot(ecdf.x, ecdf.y, color='#3182bd')
        axs[1].set_title('Empirical Distribution Function')
        axs[1].set_xlabel('Demand')
        axs[1].set_ylabel('CDF')

        plt.savefig('Test.png')

    results = comparing_distributions(
        x=demand.to_numpy(),
        family='realplus',
        orden='aic',
        verbose=False
    )

    results = pd.DataFrame(results)
#    print(results[['distribution','aic','exp_value']])

    if graph_req_ad == 1:
        fig, ax = plt.subplots(figsize=(8,5))

        plot_distribution(
            x=demand.to_numpy(),
            distribution_name=results['distribution'][0],
            ax=ax
        );

        fig, ax = plt.subplots(figsize=(8,5))

        plot_multiple_distributions(
            x=demand.to_numpy(),
            distributions_name=results['distribution'][:5],
            ax=ax
        );

    return results

#
# The following code should be active if input data is read directly from Lulus Snowflake Server
#
# conn = snowflake.connector.connect(
#     user='BPIEDRA',
#     password='Bg2020Pi',
#     account='rh02159.us-central1.gcp'
#     )
#
# conn.cursor().execute("ALTER SESSION SET QUERY_TAG = 'Testing Connection'")
# conn.cursor().execute('USE WAREHOUSE DEV_ANALYSIS')
# conn.cursor().execute('USE DATABASE LULUS')
# conn.cursor().execute('USE SCHEMA LOST_SALES')
# df = conn.cursor().execute('SELECT * FROM LOST_SALES_VIEW_NRT ORDER BY SKU_ID,BUSINESS_DAY_ID')
# datos_df = pd.DataFrame(df)
#

# The following code should be inactive if input data is read directly from Lulus Snowflake Server.  It includes reading
# input data from a .csv file.

datos = pd.read_csv('/Users/borispiedra/Version Draft Python Lulus Lost Sales/df.csv', delimiter=';',header=None, skiprows=1)
#datos = pd.read_csv('/Users/borispiedra/Documents/df19346.csv', delimiter=',',header=None, skiprows=1)
datos_df = pd.DataFrame(datos)

# Obtaining the list of SKUs.  If the characterization is going to be made for a fraction of SKU list, second line in the
# following code should be inactive.  In other case, it should include the fraction to be characterized.

SKU_ID_list = datos_df[2].unique()
SKU_ID_list = [442,5090,8818,9330,23834,57234,103594,124594,145594,181770,238282,264314,289042,309482,328138,335266,349946,366690,370386,387154,397602,425810,433306,438442,447082,478612,520992,535252,545452,548542]

prob_char_final = []
prob_char_set = []
cont = 0
for n in SKU_ID_list:
    cont += 1
    print(f"{cont}/{len(SKU_ID_list)} Fitting SKU: {n}")
    datos_SKU_ID = datos_df[datos_df[2]==n].reset_index(drop=True)

    n_sales_seasons = 0
    complete_process = 0
    current_index = 0
    a_st = 0

# Obtaining the SKU's sales seasons

    sales_seasons_limits = []
    datos_list = list(datos_SKU_ID[8])

    while complete_process == 0:
        try:
            sales_season_start = datos_list.index(True, a_st)
        except:
            break
        b_end = sales_season_start + 1
        try:
            sales_season_end = datos_list.index(False, b_end) - 1
        except:
            complete_process = 1
            sales_season_end = len(datos_list) - 1
        sales_seasons_limits.append([datos_SKU_ID[1].iloc[sales_season_start],datos_SKU_ID[1].iloc[sales_season_end],sales_season_start,sales_season_end])
        a_st = sales_season_end+1
        n_sales_seasons += 1


# Building a table that includes fiscal date, demand and availability of products for the SKU being characterized.  Main issue in
# this block is related to filling blank registers in input data.  This table is called time_demand_ava_original

    prev_prob_char = []
    n_prob_char_seas = 0
    prob_char_sku = []

    for i in range(n_sales_seasons):

        sales_season_extent = sales_seasons_limits[i][1]-sales_seasons_limits[i][0]+1
        sales_season_extent_original =  sales_seasons_limits[i][3]-sales_seasons_limits[i][2]+1
        if sales_season_extent_original>sales_season_extent:
            sales_season_extent_original = sales_season_extent
        demand_original = [0]*sales_season_extent

        for j in range(sales_seasons_limits[i][2], sales_seasons_limits[i][3]+1):
            demand_original[datos_SKU_ID[1].iloc[j]-sales_seasons_limits[i][0]] = datos_SKU_ID[7].iloc[j]

        ava_original = []
        for k in range (sales_season_extent_original-1):
            first_value = datos_SKU_ID[1].iloc[k+sales_seasons_limits[i][2]]
            second_value = datos_SKU_ID[1].iloc[k+sales_seasons_limits[i][2]+1]
            ava_original.append(datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]])
            if second_value>(first_value+1):
                if (datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]] == 0) | ((datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]] > 0) & (datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]+1] == 0)):
                    for m in range(second_value-first_value-1):
                        ava_original.append(0)
                if (datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]] > 0) & (datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]+1] > 0):
                    for m in range(second_value-first_value-1):
                        ava_original.append(datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]] - datos_SKU_ID[7].iloc[k+sales_seasons_limits[i][2]])
        ava_original.append(datos_SKU_ID[6].iloc[k+sales_seasons_limits[i][2]+1])

        time_demand_ava_original=[]
        for j in range(sales_season_extent):
            time_demand_ava_original.append([sales_seasons_limits[i][0]+j,demand_original[j],ava_original[j]])
        time_demand_ava_original = pd.DataFrame(time_demand_ava_original)


# A new table is built up from time_demand_ava_original, removing rows in which availability of products is equal to 0.
# This new table is called time_demand_ava_wouna.  Data in this table should be characterized.  Characterization is done only
# when time_demand_ava_wouna is not an empty table.

        unava_idx=time_demand_ava_original[time_demand_ava_original[2]==0].index
        time_demand_ava_wouna = time_demand_ava_original.drop(unava_idx).reset_index(drop=True)

        if len(time_demand_ava_wouna) != 0:

# If data in time_demand_ava_wouna is not enough for a probabilistic characterization process, expected value is calculated
# directly as probability of ocurrence.  A criterion is used: if sum of demand values is less than 0.5 times number of days
# of period (less than 0.5 products sold per day, as an average), it is considered that data available is not enough for
# characterization.
#
# Inside a sales season, inner seasons could exist, which probabilistic behavior is different from other seasons. If a
# time_demand_ava_wouna table has less than 60 rows, it is considered that the whole sales seasons corresponds to a unique
# inner season.  An inner season is identified by looking for valleys inside the time behavior of demand.
#
# A valley is identified by looking for persistent low values inside the data.  That means, a set of data which implies at
# least 7 days for which the weakly average of data is less than a threshold.  Threshold is taken as (Maximum Value of set data
# after outliers removal/7.5).

            if ((time_demand_ava_wouna[1].sum() / len(time_demand_ava_wouna)) < 0.5) | (sales_season_extent<=15):
                daily_prob = time_demand_ava_wouna[1].sum() / len(time_demand_ava_wouna)
                n_inner_seasons = 1
                dist_name = 'daily_dist'
                parameters = [daily_prob]
                daily_exp_val = daily_prob
                prev_prob_char.extend([sales_seasons_limits[i][0],sales_seasons_limits[i][1],daily_exp_val])
                n_prob_char_seas += 1
            else:
                if len(time_demand_ava_wouna)<=29:
                    n_inner_seasons = 1
                    demand_final = time_demand_ava_wouna[1]
                    exp_value = time_demand_ava_wouna[1].sum() / len(time_demand_ava_wouna)
                    kon = 0
                    prev_prob_char.extend([sales_seasons_limits[i][0], sales_seasons_limits[i][1], exp_value])
                    n_prob_char_seas += 1
                else:

                    demand_nv = time_demand_ava_wouna[1]
                    max_val_ref, time_demand_ava_woout = out_rem(demand_nv, time_demand_ava_wouna)
                    weekly_ave_demand = [time_demand_ava_wouna[1].iloc[k:k + 7].mean() for k in range(len(time_demand_ava_wouna) - 6)]
                    weekly_ave_demand_df = pd.DataFrame(weekly_ave_demand)
                    lowest_idx = weekly_ave_demand_df[weekly_ave_demand_df[0] < (max_val_ref / 7.5)].index
                    mark = 0
                    n_limits=0
                    if len(lowest_idx)>0:
                        warn = lowest_idx[0]
                        counter = 0
                        season_found = 0
                        n_inner_seasons = 0

                        inner_seasons_limits = []
                        for k in range(1, len(lowest_idx)):
                            if lowest_idx[k] == lowest_idx[k - 1] + 1:
                                counter += 1
                                if (k==(len(lowest_idx)-1)) & (counter>6):
                                    inner_season_end = lowest_idx[k]
                                    counter=0
                                    season_found=0
                                    n_inner_seasons += 1
                                    warn = lowest_idx[k]
                                    inner_seasons_limits.append([inner_season_start, inner_season_end])
                                else:
                                    if counter == 6:
                                        inner_season_start = warn
                                        season_found = 1
                            else:
                                if season_found == 1:
                                    inner_season_end = lowest_idx[k - 1]
                                    counter = 0
                                    season_found = 0
                                    n_inner_seasons += 1
                                    warn = lowest_idx[k]
                                    inner_seasons_limits.append([inner_season_start, inner_season_end])
                                else:
                                    warn = lowest_idx[k]
                                    counter = 0
                    else:
                        mark = 1

                    if mark==0:
                        inner_seasons_limits_series = pd.DataFrame(inner_seasons_limits)

                        n_remove = 0
                        for k in range(1, n_inner_seasons):
                            if inner_seasons_limits_series[0][k - n_remove] < (
                                    inner_seasons_limits_series[1][k - n_remove - 1] + 30):
                                a_start = inner_seasons_limits_series[0][k - n_remove - 1]
                                b_end = inner_seasons_limits_series[1][k - n_remove]
                                inner_seasons_limits_series[0][k - n_remove - 1] = a_start
                                inner_seasons_limits_series[1][k - n_remove - 1] = b_end
                                inner_seasons_limits_series = inner_seasons_limits_series.drop(k - n_remove).reset_index(
                                    drop=True)
                                n_remove += 1
                                n_inner_seasons -= 1

                        inner_seasons_limits_final = []
                        n_limits = 0
                        for k in range(n_inner_seasons):
                            n_limits += 1
                            inner_seasons_limits_final.append(inner_seasons_limits_series[0].iloc[k] + int(
                                (inner_seasons_limits_series[1].iloc[k] - inner_seasons_limits_series[0].iloc[k]) / 2) + 1)

                    inner_seasons = []
                    inner_seasons_local = []
                    if n_limits > 0:
                        for k in range(n_limits):
                            if k == 0:
                                inner_seasons.append(
                                    [sales_seasons_limits[i][0], time_demand_ava_wouna[0].iloc[inner_seasons_limits_final[k]]])
                                inner_seasons_local.append([0, inner_seasons_limits_final[k]])
                            else:
                                inner_seasons.append([time_demand_ava_wouna[0].iloc[inner_seasons_limits_final[k - 1]] + 1,
                                                time_demand_ava_wouna[0].iloc[inner_seasons_limits_final[k]]])
                                inner_seasons_local.append([inner_seasons_limits_final[k - 1] + 1, inner_seasons_limits_final[k]])
                        inner_seasons.append(
                            [time_demand_ava_wouna[0].iloc[inner_seasons_limits_final[n_limits - 1]] + 1, sales_seasons_limits[i][1]])
                        inner_seasons_local.append([inner_seasons_limits_final[n_limits - 1] + 1, len(time_demand_ava_wouna) - 1])
                    else:
                        inner_seasons.append([sales_seasons_limits[i][0],sales_seasons_limits[i][1]])
                        inner_seasons_local.append([0,len(time_demand_ava_wouna)-1])

                    for k in range(n_limits + 1):
                        data_in = time_demand_ava_wouna[1].iloc[
                                inner_seasons_local[k][0]:inner_seasons_local[k][1] + 1].reset_index(drop=True)
                        complete_data_in = time_demand_ava_wouna.iloc[
                                        inner_seasons_local[k][0]:inner_seasons_local[k][1] + 1].reset_index(drop=True)
                        max_val_ref, complete_data_in_woout = out_rem(data_in, complete_data_in)
                        data_in_2 = complete_data_in_woout[1]

                        if len(data_in_2) >= 30:
                            char_results = prob_char_1sku(data_in_2, 0, 0, 0)
                            exp_value = char_results['exp_value'][0]
                            max_value = data_in_2.max()
                            if exp_value>max_value:
                                exp_value = data_in_2.sum()/len(data_in_2)
                        else:
                            exp_value = data_in_2.sum() / len(data_in_2)
                        kon = 0
                        while (exp_value < 0) | (exp_value > 100) | (np.isnan(exp_value)) | (math.isinf(exp_value)):
                            kon += 1
                            exp_value = char_results['exp_value'][kon]
                        prev_prob_char.extend([inner_seasons[k][0], inner_seasons[k][1], exp_value])
                        n_prob_char_seas += 1

# Building the dataframe that contains SKUs probabilistic characterization.  This dataframe is saved in a .csv file

    prob_char_sku.extend([n, n_prob_char_seas])
    prob_char_sku.extend(prev_prob_char)
    prob_char_set.append(prob_char_sku)
    print(prob_char_set)

    prob_char_set_df = pd.DataFrame(prob_char_set)
    prob_char_set_df.to_csv('/Users/borispiedra/Version Draft Python Lulus Lost Sales/Prob_Char_Set_LSTesting_nv.csv', index=False)
