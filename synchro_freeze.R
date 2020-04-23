conv_str2list <- function(df) {
    # Convert strings to integer list
    for (colname in str_subset(names(df), "lagt_*")){
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