order1_logrithm = function( data_set){
        tmp = subset( data_set, order == 1 )
        
        tmp$log_revenue = log(tmp$revenue)
        tmp$loo_mean_revenue_by_pid = abs(tmp$loo_mean_revenue_by_pid)
        
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
        
        tmp[,need_to_log] = log(tmp[,need_to_log])
        return(tmp)
}



plus1_logrithm = function( data_set){
       
        tmp$log_revenue = log(1+ tmp$revenue)
        tmp$loo_mean_revenue_by_pid = abs(1+tmp$loo_mean_revenue_by_pid)
        
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
        
        tmp[,need_to_log] = log(1+ tmp[,need_to_log])
        
        return(tmp)
}

