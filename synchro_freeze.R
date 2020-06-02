#############################################################################################################
conv_str2list <- function(df,rex) {
    library(stringr)
    
    # Convert strings to integer list
    for (colname in str_subset(names(df), rex)){
        # Generate working column
        w_colname = paste0("w_", colname)
        df[,w_colname] = NA

        # Store the generated lists in the new column
        for (i in c(1:nrow(df))){
            df[[i,w_colname]] = list(as.integer(unlist(strsplit(gsub(" +",',',gsub("\\[ *|\\]|\\'",'',df[i,colname])),','))))
        }

        # delete old column
        df = df[,!(names(df) %in% c(colname))]

        # rename working column to original name
        names(df)[names(df) == w_colname] <- colname    
    }
    return(df)
}

#############################################################################################################
#**Set boxplot showing ymin, lower SEM, mean, upper SEM, and ymax**<BR>
#https://stackoverflow.com/questions/25999677/how-to-plot-mean-and-standard-error-in-boxplot-in-r

MinMeanSEMMax <- function(x) {
  v <- c(min(x), mean(x) - sd(x)/sqrt(length(x)), mean(x), mean(x) + sd(x)/sqrt(length(x)), max(x))
  names(v) <- c("ymin", "lower", "middle", "upper", "ymax")
  v
}

# Modify above function to remove min and max wikskers
MeanSEM <- function(x) {
  v <- c(mean(x) - sd(x)/sqrt(length(x)), mean(x) - sd(x)/sqrt(length(x)), 
         mean(x),
         mean(x) + sd(x)/sqrt(length(x)), mean(x) + sd(x)/sqrt(length(x)))
  names(v) <- c("ymin", "lower", "middle", "upper", "ymax")
  v
}

MeanSD <- function(x) {
  v <- c(mean(x) - sd(x), mean(x) - sd(x), 
         mean(x),
         mean(x) + sd(x), mean(x) + sd(x))
  names(v) <- c("ymin", "lower", "middle", "upper", "ymax")
  v
}
#############################################################################################################
dis_summary <- function(df.input){
    #######################################
    # Display summary
    cat("dimension: ",dim(df.input),'\n')
    #str(df.input)
    print(sapply(df.input, class))
    #sapply(df.input, typeof)
    head(df.input,2)
    #######################################
}
#############################################################################################################
# Save data frame as df and Excel file
save_files <-function(df.input,path,base){
    
    # Save DF (.Rda)
    base1 = paste0(base,".Rda")
    filename = file.path(path, base1)
    save(df.input, file=filename)

    # Output as Excel file, not included with list (.xlsx)
    base2 = paste0(base,".xlsx")    
    filename = file.path(path, base2)
    write_xlsx(df.input,filename)
}
#############################################################################################################