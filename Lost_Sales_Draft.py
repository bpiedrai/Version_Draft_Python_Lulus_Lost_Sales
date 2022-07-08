import pandas as pd

#
# Reading input sales data from a .csv file.   SKU_list includes the list of SKUs for lost sales calculation.  It it is
# not included, it takes all SKUs available in input data file.
#

datos = pd.read_csv('/Users/borispiedra/447082_mod4.csv', delimiter=';', header=None)
datos_df = pd.DataFrame(datos)
SKU_list = datos_df[2].unique()
SKU_list = [447082]


#
# Reading of probabilistic characterization data from a .csv file.  This file was created with the Probabilistic Characterization
# Module
#


datos_prob_char = pd.read_csv('/Users/borispiedra/Version Draft Python Lulus Lost Sales/Prob_Char_Set_LSTesting_nv3_LV_7.5_5.25_staave.csv', delimiter=',',index_col=False)
datos_prob_char_df = pd.DataFrame(datos_prob_char)

#
# Definition of the time period for lost sales calculation
#

initial_date = 5217
final_date = 5336


lost_sales_set = []
lost_sales_SKU = 0

#
# Calculation of lost sales for each sku in SKU_list
#



for i in SKU_list:
    ind_SKU = datos_df[datos_df[2]==i].index
    datos_SKU = datos_df.iloc[ind_SKU].reset_index(drop=True)

    extent_SKU = datos_SKU[1].iloc[len(datos_SKU) - 1] - datos_SKU[1].iloc[0] + 1

    demand_original = [0] * extent_SKU

    SKU_initial_date = datos_SKU[1][0]
    SKU_final_date = datos_SKU[1][len(datos_SKU) - 1]

#
# Building a table which includes fiscal date, demand and availability of products for each SKU in SKU_list.  Esta tabla
# es denominada time_demand_ava
#

    for j in range(len(datos_SKU)):
        demand_original[datos_SKU[1][j] - SKU_initial_date] = datos_SKU[7][j]

    ava_original = []
    for k in range(len(datos_SKU) - 1):
        first_value = datos_SKU[1][k]
        second_value = datos_SKU[1][k + 1]
        ava_original.append(datos_SKU[6][k])
        if second_value > (first_value + 1):
            if (datos_SKU[6][k] == 0) | ((datos_SKU[6][k] > 0) & (datos_SKU[6][k + 1] == 0)):
                for m in range(second_value - first_value - 1):
                    ava_original.append(0)
            if (datos_SKU[6][k] > 0) & (datos_SKU[6][k + 1] > 0):
                for m in range(second_value - first_value - 1):
                    ava_original.append(datos_SKU[6][k] - datos_SKU[7][k])
    ava_original.append(datos_SKU[6][k + 1])

    time_demand_ava = []
    for j in range(extent_SKU):
        time_demand_ava.append([SKU_initial_date + j, demand_original[j], ava_original[j]])
    time_demand_ava = pd.DataFrame(time_demand_ava)

#
# Calculation of lost sales.   Presales are substracted from lost sales
#

    n_index = datos_prob_char_df[datos_prob_char_df['0'] == i].index
    n_index = n_index[0]
    if datos_prob_char_df['1'][n_index] == 0:
        lost_sales_SKU = 0
    else:
        if (initial_date < datos_prob_char_df['2'][n_index]) & (final_date < datos_prob_char_df['2'][n_index]):
            lost_sales_SKU = 0
        else:
            lost_sales_SKU = 0
            for j in range(datos_prob_char_df['1'][n_index]):
                a = 3 * (j + 1) - 1
                b = a + 1
                c = b + 1
                a = str(a)
                b = str(b)
                c = str(c)
                ai = int(datos_prob_char_df[a][n_index])
                bi = int(datos_prob_char_df[b][n_index])
                if (initial_date in range(ai, bi + 1)) & (final_date in range(ai, bi + 1)):
                    df_red = time_demand_ava.loc[
                                initial_date - SKU_initial_date:final_date - SKU_initial_date].reset_index(drop=True)
                    ind_reg = df_red[df_red[2] == 0].index
                    count_plus = len(ind_reg) * datos_prob_char_df[c][n_index]
                    count_minus = df_red[1].iloc[ind_reg].sum()
                    lost_sales_SKU = lost_sales_SKU + count_plus - count_minus

                if (initial_date in range(ai, bi + 1)) & (final_date > bi):
                    df_red = time_demand_ava.loc[initial_date - SKU_initial_date:bi - SKU_initial_date].reset_index(
                        drop=True)
                    ind_reg = df_red[df_red[2] == 0].index
                    count_plus = len(ind_reg) * datos_prob_char_df[c][n_index]
                    count_minus = df_red[1].iloc[ind_reg].sum()
                    lost_sales_SKU = lost_sales_SKU + count_plus - count_minus

                if (initial_date < ai) & (final_date in range(ai, bi + 1)):
                    df_red = time_demand_ava.loc[ai - SKU_initial_date:final_date - SKU_initial_date].reset_index(
                        drop=True)
                    ind_reg = df_red[df_red[2] == 0].index
                    count_plus = len(ind_reg) * datos_prob_char_df[c][n_index]
                    count_minus = df_red[1].iloc[ind_reg].sum()
                    lost_sales_SKU = lost_sales_SKU + count_plus - count_minus

                if (initial_date < ai) & (final_date > bi):
                    df_red = time_demand_ava.loc[ai - SKU_initial_date:bi - SKU_initial_date].reset_index(drop=True)
                    ind_reg = df_red[df_red[2] == 0].index
                    count_plus = len(ind_reg) * datos_prob_char_df[c][n_index]
                    count_minus = df_red[1].iloc[ind_reg].sum()
                    lost_sales_SKU = lost_sales_SKU + count_plus - count_minus

#
# Building of the Lost Sales DataFrame, which is saved in a .csv file
#

    lost_sales_set.append([i, round(lost_sales_SKU)])

lost_sales_set = pd.DataFrame(lost_sales_set)
print('lost_sales_set=', lost_sales_set)
lost_sales_set.to_csv('/Users/borispiedra/Version Draft Python Lulus Lost Sales/Lost_Sales_LSTesting_447082_P1_t4_staave.csv')




