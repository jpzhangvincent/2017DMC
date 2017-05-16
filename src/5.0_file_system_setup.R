## this is the file to set up the necessary folder system in order to run your code
## if you don't have your folder system set up

system("mkdir DMC")
system("mkdir DMC/data")
system("ln -s /scratch/2017dmc ./DMC/data/processed")
system("mkdir DMC/data/preds1stLevel")
system("mkdir DMC/src")
system("mkdir DMC/models")
system("mkdir DMC/models/1stLevel")

## copy your code.r to the DMC/src folder then run the following
## R CMD BATCH code.r &
