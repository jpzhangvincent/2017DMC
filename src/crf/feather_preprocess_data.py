# -*- coding: utf-8 -*-

import pycrfsuite
import pandas as pd
import pickle
import feather as ft





# Define feature extractor
def row2features(row):
    """
    row is a pandas dataframe row
    no continuous features
    not taking price fluctuation into account
    not using content and unit features
    not using reference retail price (rrp)
    """
    features = [
        'bias',
        'content={}'.format(row['content']),
        'pharmForm={}'.format(row['pharmForm']),
        'unit={}'.format(row['unit']),
        'category={}'.format(row['category']),
        'group={}'.format(row['group']),
        'manufacturer={}'.format(row['manufacturer']),
        'pid={}'.format(row['pid']),
        'day_mod_7={}'.format(row['day_mod_7']),
        'lineID={}'.format(row['lineID']),
        'day={}'.format(row['day']),
        'adFlag={}'.format(row['adFlag']),
        'availability={}'.format(row['availability']),
        'price={}'.format(row['price']),
        'genericProduct={}'.format(row['genericProduct']),
        'salesIndex={}'.format(row['salesIndex']),
        'campaignIndex={}'.format(row['campaignIndex']),
        'rrp={}'.format(row['rrp']),
        'num_pid_click={}'.format(row['num_pid_click']),
        'mean_pid_click={}'.format(row['mean_pid_click']),
        'num_pid_basket={}'.format(row['num_pid_basket']),
        'mean_pid_basket={}'.format(row['mean_pid_basket']),
        'num_pid_order={}'.format(row['num_pid_order']),
        'mean_pid_order={}'.format(row['mean_pid_order']),
        'buy_one_prob={}'.format(row['buy_one_prob']),
        'buy_more_prob={}'.format(row['buy_more_prob']),
        'day_mod_10={}'.format(row['day_mod_10']),
        'day_mod_14={}'.format(row['day_mod_14']),
        'day_mod_28={}'.format(row['day_mod_28']),
        'day_mod_30={}'.format(row['day_mod_30']),
        'pharmForm_isNA={}'.format(row['pharmForm_isNA']),
        'category_isNA={}'.format(row['category_isNA']),
        'campaignIndex_isNA={}'.format(row['campaignIndex_isNA']),
        'competitorPrice_isNA={}'.format(row['competitorPrice_isNA']),
        'competitorPrice_imputed={}'.format(row['competitorPrice_imputed']),
        # 'content_cnt={}'.format(row['content_cnt']),
        # 'group_cnt={}'.format(row['group_cnt']),
        # 'manufacturer_cnt={}'.format(row['manufacturer_cnt']),
        # 'unit_cnt={}'.format(row['unit_cnt']),
        # 'pharmForm_cnt={}'.format(row['pharmForm_cnt']),
        # 'category_cnt={}'.format(row['category_cnt']),
        # 'campaignIndex_cnt={}'.format(row['campaignIndex_cnt']),
        # 'salesIndex_cnt={}'.format(row['salesIndex_cnt']),
        # 'cnt_click_byday7={}'.format(row['cnt_click_byday7']),
        # 'cnt_basket_byday7={}'.format(row['cnt_basket_byday7']),
        # 'cnt_order_byday7={}'.format(row['cnt_order_byday7']),
        # 'price_diff={}'.format(row['price_diff']),
        'islower_price={}'.format(row['islower_price']),
        'price_discount={}'.format(row['price_discount']),
        'is_discount={}'.format(row['is_discount']),
        'competitor_price_discount={}'.format(row['competitor_price_discount']),
        # 'price_discount_diff={}'.format(row['price_discount_diff']),
        'isgreater_discount={}'.format(row['isgreater_discount']),
        'max_price_disc={}'.format(row['max_price_disc']),
        'min_price_disc={}'.format(row['min_price_disc']),
        'var_price_disc={}'.format(row['var_price_disc']),
        'p25_price_disc={}'.format(row['p25_price_disc']),
        'median_price_disc={}'.format(row['median_price_disc']),
        'p75_price_bygroup={}'.format(row['p75_price_bygroup']),
        'last_price={}'.format(row['last_price']),
        # 'lprice_chg_pct={}'.format(row['lprice_chg_pct']),
        # 'last3_price_avg={}'.format(row['last3_price_avg']),
        # 'last3_price_min={}'.format(row['last3_price_min']),
        # 'last3_price_max={}'.format(row['last3_price_max']),
        # 'pid_ref={}'.format(row['pid_ref']),
        # 'manufacturer_ref={}'.format(row['manufacturer_ref']),
        # 'group_ref={}'.format(row['group_ref']),
        # 'category_ref={}'.format(row['category_ref']),
        # 'unit_ref={}'.format(row['unit_ref']),
        # 'pharmForm_ref={}'.format(row['pharmForm_ref']),
        # 'content_ref={}'.format(row['content_ref']),
        # 'content_part1={}'.format(row['content_part1']),
        # 'content_part2={}'.format(row['content_part2']),
        # 'content_part3={}'.format(row['content_part3']),
        # 'total_units={}'.format(row['total_units']),
        # 'rrp_per_unit={}'.format(row['rrp_per_unit']),
        # 'price_per_unit={}'.format(row['price_per_unit']),
        # 'competitorPrice_per_unit={}'.format(row['competitorPrice_per_unit'])

    ]
    
    return features


print('reading data')
train = ft.read_dataframe('/data/hugo/train_may_1st.feather')
test = ft.read_dataframe('/data/hugo/validation_may_1st.feather')

# Feature extraction
print('extracting features and labels')
X_train = [row2features(row) for _, row in train.iterrows()]
y_train = [str(row['order']) for _, row in train.iterrows()]
X_test = [row2features(row) for _, row in test.iterrows()]
y_test = [str(row['order']) for _, row in test.iterrows()]



# print('splitting training and testing sets')
# # 1708473 is the first item of the 61st day
# X_train = X[:1708473]
# X_test = X[1708473:]
# y_train = y[:1708473]
# y_test = y[1708473:]

# Save processed data
# print('saving preprocessed data')
with open('X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)
with open('y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
with open('X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)



