library(feather)
library(ggplot2)
library(readr)

training_set   = read_feather('training_set.feather')
validation_set = read_feather('validation_set.feather')

all_set = rbind(training_set,validation_set) 

all_set$lineID = as.numeric(all_set$lineID)
all_set =as.data.frame( all_set[order(all_set$lineID),])


#colnames(all_set)
#sapply(all_set,class)

categorical_feathers_name = c('lineID', 'day', 'click', 'basket', 'order', 'revenue', 'group', 'content',
                         'unit', 'availability', 'adFlag', 'pharmForm', 'category', 'manufacturer',
                         'pid', 'day_mod_7', 'day_mod_30', 'day_mod_10', 'genericProduct', 
                         'salesIndex','campaignIndex', 'day_mod_14', 'day_mod_28', 'pharmForm_isNA',
                         'category_isNA', 'campaignIndex_isNA', 'competitorPrice_isNA',
                         "group_beginNum", "content_part1", "content_part2", "content_part3","total_units",
                         'islower_price', 'is_discount', 'last_avaibility', 'last_adFlag',
                         "avaibility_transition",  "adFlag_transition", 'isgreater_discount',
                         'avgnext5_price_isLower','avglast5_price_isLower'
                         )


categorical_feathers = all_set[,categorical_feathers_name]
shouldbe_factorize= c( 'day', 'click', 'basket', 'order', 'group', 'content',
                       'unit', 'availability', 'adFlag', 'pharmForm', 'category', 'manufacturer',
                       'pid', 'day_mod_7', 'day_mod_30', 'day_mod_10', 'genericProduct', 
                       'salesIndex','campaignIndex', 'day_mod_14', 'day_mod_28', 'pharmForm_isNA',
                       'category_isNA', 'campaignIndex_isNA', 'competitorPrice_isNA',
                       "group_beginNum", "content_part1", "content_part2", "content_part3","total_units",
                       'islower_price', 'is_discount', 'last_avaibility', 'last_adFlag',
                       "avaibility_transition",  "adFlag_transition", 'isgreater_discount',
                       'avgnext5_price_isLower','avglast5_price_isLower'
)

for(f in 1:length(shouldbe_factorize)){
        categorical_feathers[,shouldbe_factorize[f]] = factor(categorical_feathers[,shouldbe_factorize[f]])
}


contin_feathers = all_set[, !( colnames(all_set)%in% categorical_feathers_name) ]

contin_feathers$last_price_diff = all_set$last_price - all_set$price
contin_feathers$last5_price_avg_price_diff = all_set$last5_price_avg - all_set$price
contin_feathers$last5_price_min_price_diff = all_set$last5_price_min - all_set$price
contin_feathers$last5_price_max_price_diff = all_set$last5_price_max - all_set$price
contin_feathers$next_price_price_diff =  all_set$next_price -  all_set$price
contin_feathers$next5_price_avg_price_diff =  all_set$next5_price_avg -  all_set$price
contin_feathers$next5_price_max_price_diff = all_set$next5_price_max -  all_set$price
contin_feathers$next5_price_min_price_diff =  all_set$next5_price_min -  all_set$price

contin_feathers = contin_feathers[, !colnames(contin_feathers) %in% c('last_price', 'last5_price_avg',
                                                                      'last5_price_min','last5_price_max',
                                                                      'next5_price_avg', 'next5_price_max',
                                                                      'next5_price_min','next_price')]
num_levels =10
discretized_feathers = contin_feathers
colnames(discretized_feathers) = paste('discrete_', colnames(contin_feathers),sep='')

for(i in 75:ncol(contin_feathers) ){
        cat(i)
        a_feather =  contin_feathers[,i]
        sam_feather = sample(a_feather , 20000, replace = FALSE) 
        sam_feather = sam_feather[sam_feather <1e10]
        sam_hclust  = hclust(dist(sam_feather))
        labels_sam= cutree(sam_hclust, k = num_levels)
        breaks_raw= sapply(1:num_levels, function(i) range(sam_feather[labels_sam==i]))
        breaks = unique(c(-1e10, breaks_raw[1,1], 
                  (breaks_raw[1,2:num_levels] + breaks_raw[2,1:(num_levels-1)])/2, 
                  breaks_raw[2,num_levels], 1e10 ))
        discretized_feathers[,i] = cut(a_feather, breaks, ordered_result = TRUE)
}


binded_discretized_feathers=cbind(categorical_feathers, discretized_feathers)

discrete_training_set= subset(binded_discretized_feathers, day <=77)


#tmp1 = paste('discrete_',colnames(contin_feather), '.csv',sep='')
#tmp2 = list.files()
#tmp1[!tmp1%in%tmp2]


#colnames(training_set)[!colnames(training_set) %in% c(colnames(contin_feather), colnames(categorical_feathers))]
