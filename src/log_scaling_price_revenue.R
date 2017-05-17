offset_logrithm = function( data_set){
        # create another revenue column called log_revenue
        data_set$log_revenue = log(1e-5+ data_set$revenue) - log(1e-5)
        data_set$loo_mean_revenue_by_pid = abs(1e-5+data_set$loo_mean_revenue_by_pid) - log(1e-5)
        
       # all columns need to transfer to log-scale
        need_to_log=c("avg_price_basket_info",
                      "avg_price_click_info",
                      "avg_price_order_info",
                      "avg_revenue_by_group_10",
                      "avg_revenue_by_group_30",
                      "avg_revenue_by_group_7",
                      "competitorPrice_per_unit",
                      "loo_mean_revenue_by_pid",
                      "next_price",
                      "next5_price_avg",
                      "next5_price_max",
                      "next5_price_min",
                      "prev_price",
                      "prev5_price_avg",
                      "prev5_price_diff",
                      "prev5_price_max",
                      "prev5_price_min",
                      "price",
                      "price_per_unit",
                      "rrp",
                      "rrp_per_unit")
        
        data_set[,need_to_log] = log(1e-5+ data_set[,need_to_log]) - log(1e-5)
        
        return(data_set)
}


