setwd("~/Dropbox/uc_davis/dmc_2017")
train_items = read.csv("train_items.csv")
items = read.csv("items.csv",sep='|')
train_price_per_unit = read.csv("train_price_per_unit.csv")


head(train_items)
#manufacturer
#number of PIDs
#number of lines
#value of products
#vs adFlag
#vs price per unit


################# about manufacturer
# number of PIDs and lineID
manuf_unq_it = unique(items$manufacturer) #1067
manuf_unq_tr = unique(train_items$manufacturer) #1065
manuf = data.frame(table(items$manufacturer))
manuf$line = 0
lst = 1:1067
numberoflines = lapply(lst, function(i) manuf$line[i] = sum(train_items$manufacturer == manuf$Var1[i]))
manuf$line = unlist(numberoflines)

# adFlag
foruse = data.frame(train_items$manufacturer,train_items$adFlag)
colnames(foruse)
library(dplyr)
manuf_adF = summarise(group_by(foruse, train_items.manufacturer),mean(train_items.adFlag))

manuf$adFlag = 0
for(i in lst){
  if(sum(manuf_adF$train_items.manufacturer == i) != 0)
    manuf$adFlag[i] = manuf_adF$`mean(train_items.adFlag)`[which(manuf_adF$train_items.manufacturer == i)]
}


# price per unit and competitor price per unit
train_price_per_unit$manufacturer = train_items$manufacturer
manuf_price_per_unit = summarise(group_by(train_price_per_unit, manufacturer), mean(price_per_unit))
manuf_competitor_price_per_unit = summarise(group_by(train_price_per_unit, manufacturer), mean(competitor_price_per_unit, na.rm = TRUE))

manuf$price_per_unit = 0
manuf$competitor_price_per_unit = 0
for(i in lst){
  if(sum(manuf_price_per_unit$manufacturer == i) != 0){
    manuf$price_per_unit[i] = manuf_price_per_unit$`mean(price_per_unit)`[which(manuf_price_per_unit$manufacturer == i)]
    manuf$competitor_price_per_unit[i] = manuf_competitor_price_per_unit$`mean(competitor_price_per_unit, na.rm...`[which(manuf_price_per_unit$manufacturer == i)]
  }
}


#k means 
set.seed(27)

km5 = kmeans(manuf[,-c(1,5)], 5) #95.8%
manuf$cluster5 = km5$cluster

km7 = kmeans(manuf[,-c(1,5)], 7) #96.9%
manuf$cluster7 = km7$cluster

km = kmeans(manuf[,-1], 10)
manuf$cluster10 = km$cluster
#manuf[(manuf$cluster == 1),][1:50,] #97.1%


colnames(manuf) = c("manufacturer","Number of PIDs", "Number of lineIDs", "Percent of adFlag", "cluster5", "cluster7", "cluster10")
write.csv(manuf, "manufacturer_Olivia.csv", row.names = FALSE)

################### about group
unique_group_order1 = unique(train_items$group[which(train_items$order == 1)])
unique_group_order0 = unique(train_items$group[which(train_items$order == 0)])

table_group_order1_most = sort(table(train_items$group[which(train_items$order == 1)]), decreasing = TRUE)[1:100]
table_group_order0_least = sort(table(train_items$group[which(train_items$order == 0)]), decreasing = FALSE)[1:100]
df_group_order1_most_order0_least = data.frame(table_group_order1_most,table_group_order0_least)
colnames(df_group_order1_most_order0_least) = c("order=1", "most100", "order=0", "least100")
df_group_order1_most_order0_least$`order=1` = as.character(df_group_order1_most_order0_least$`order=1`)
df_group_order1_most_order0_least$`order=0` = as.character(df_group_order1_most_order0_least$`order=0`)

for(i in 1:100){
  if(sum(df_group_order1_most_order0_least$`order=0` == df_group_order1_most_order0_least$`order=1`[i]) != 0){
    print(df_group_order1_most_order0_least$`order=1`[i])
  }
}


table_group_order0_most = sort(table(train_items$group[which(train_items$order == 0)]), decreasing = TRUE)[1:100]
table_group_order1_least = sort(table(train_items$group[which(train_items$order == 1)]), decreasing = FALSE)[1:100]
df_group_order0_most_order1_least = data.frame(table_group_order0_most, table_group_order1_least)
colnames(df_group_order0_most_order1_least) = c("order=0", "most100", "order=1", "least100")
df_group_order0_most_order1_least$`order=0` = as.character(df_group_order0_most_order1_least$`order=0`)
df_group_order0_most_order1_least$`order=1` = as.character(df_group_order0_most_order1_least$`order=1`)


for(i in 1:100){
  if(sum(df_group_order0_most_order1_least$`order=0` == df_group_order0_most_order1_least$`order=1`[i]) != 0){
    print(df_group_order0_most_order1_least$`order=1`[i])
  }
}


n = 0
for(i in 1:100){
  if(sum(df_group_order0_most_order1_least$`order=0` == df_group_order1_most_order0_least$`order=1`[i]) != 0){
    #print(df_group_order1_most_order0_least$`order=1`[i])
    n = n+1
  }
}
n

#### to see which groups in order =1 most 100 but not in order=0 most 100
nn = 0
ls = list()
for(i in 1:100){
  if(sum(df_group_order0_most_order1_least$`order=0` == df_group_order1_most_order0_least$`order=1`[i]) == 0){
    ls[i] = df_group_order1_most_order0_least$`order=1`[i]
    nn = nn+1
  }
}
nn
ls

#[1] "21OJ1"    "13OJ11OK" "10OZ03OI" "1COJ00OQ" "18OI01OS" "10OZ03OH" "12OS0F"   "18OZ00IK" "10OZ12OI" "10OI01OK" "22OI4"    "12OS2F"   "1COZ00OK" "1COS20IZ"
#[15] "10OK31OH" "19OZ22OZ" "1AOK1F"   "13OH0F"   "10OZ12OZ" "2E"       "22OIK" 



### 21 in order = 1 most 100 not in order = 0 most 100
df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "21OJ1"),1:2]
#   order=1 most100
#48   21OJ1    3397
                                                                        #      order = 0    order = 1
sum(train_items$group[which(train_items$order == 0)] == "21OJ1") #4683  #total   2050913       705090


df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "13OJ11OK"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "13OJ11OK")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OZ03OI"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OZ03OI")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "1COJ00OQ"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "1COJ00OQ")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "18OI01OS"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "18OI01OS")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OZ03OH"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OZ03OH")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "12OS0F"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "12OS0F")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "18OZ00IK"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "18OZ00IK")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OZ12OI"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OZ12OI")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OI01OK"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OI01OK")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "22OI4"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "22OI4")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "12OS2F"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "12OS2F")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "1COZ00OK"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "1COZ00OK")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "1COS20IZ"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "1COS20IZ")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OK31OH"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OK31OH")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "19OZ22OZ"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "19OZ22OZ")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "1AOK1F"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "1AOK1F")

#"13OH0F"   "10OZ12OZ" "2E"       "22OIK" 
df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "13OH0F"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "13OH0F")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "10OZ12OZ"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "10OZ12OZ")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "2E"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "2E")

df_group_order1_most_order0_least[which(df_group_order1_most_order0_least$`order=1` == "22OIK"),1:2]
sum(train_items$group[which(train_items$order == 0)] == "22OIK")


#### to see which groups in order =0 most 100 but not in order=1 most 100
nnn = 0
lss = list()
for(i in 1:100){
  if(sum(df_group_order1_most_order0_least$`order=1` == df_group_order0_most_order1_least$`order=0`[i]) == 0){
    lss[i] = df_group_order0_most_order1_least$`order=0`[i]
    nnn = nnn+1
  }
}
nnn
lss
unlist(lss)
#[1] "1COS2"    "23"       "20OHE"    "20OIF"    "19OJ0FZS" "21OK0"    "14OH1"    "12OS10OK" "13OI0EIJ" "20OHK"    "21OS0"    "20OH0"    "22OIG"    "19OS2F"  
#[15] "22OI9"    "24OZ"     "21OKF"    "10OX01"   "14OH2F"   "22OZG"    "10OTF0OI"


for(i in 1:21){
  print(df_group_order0_most_order1_least[which(df_group_order0_most_order1_least$`order=0` == unlist(lss)[i]),1:2])
  print(sum(train_items$group[which(train_items$order == 1)] == unlist(lss)[i]))
}


############# about availability
avaiability = train_items[train_items$availability == 4,]
nrow(avaiability) #10344
availability_4_unique_pid = unique(avaiability$pid)
length(unique(avaiability$pid)) #314

avblt_not4_pid = train_items[train_items$availability != 4,4]

listtt = list()
for(i in 1:314){
  summm = sum(avblt_not4_pid == availability_4_unique_pid[i])
  listtt[i] = summm
}
sum(unlist(listtt) == 0) #129
sum(avaiability$order)/10344 #0.0002900232



avaiability_3 = train_items[train_items$availability == 3,]
nrow(avaiability_3) #44893
availability_3_unique_pid = unique(avaiability_3$pid)
length(availability_3_unique_pid) #4304

avblt_not3_pid = train_items[train_items$availability != 3,4]

listtt3 = list()
for(i in 1:4304){
  summm = sum(avblt_not3_pid == availability_3_unique_pid[i])
  listtt3[i] = summm
}
unlist3 = unlist(listtt3)
sum(unlist3 == 0)#1289
sum(avaiability_3$order)/44893 #0.1036019





avaiability_2 = train_items[train_items$availability == 2,]
nrow(avaiability_2) #185194
availability_2_unique_pid = unique(avaiability_2$pid)
length(availability_2_unique_pid) #11695

avblt_not2_pid = train_items[train_items$availability != 2,4]

countt = 0
for(i in 1:11695){
  if(sum(avblt_not2_pid == availability_2_unique_pid[i]) == 0){
    countt = countt + 1
  }
}
countt #5683
sum(avaiability_2$order) #26980
26980/185194 #0.1456851


avaiability_1 = train_items[train_items$availability == 1,]
nrow(avaiability_1) #2515572
availability_1_unique_pid = unique(avaiability_1$pid)
length(availability_1_unique_pid) #13052

avblt_not1_pid = train_items[train_items$availability != 1,4]

countt1 = 0
for(i in 1:13052){
  if(sum(avblt_not1_pid == availability_1_unique_pid[i]) == 0){
    countt1 = countt1 + 1
  }
}
countt1 #7910
sum(avaiability_1$order) #673456
673456/2515572 #0.2677



