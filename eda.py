###############################
## Histogram by target variable:
###############################

def hist_class(category, df, width, fig_width, fig_heigt):
    
    one = df[df[category]==1]
    zero = df[df[category]==0]
    
    non_obj_cols = [col for col in df.columns if df[col].dtypes != 'O']
    non_obj_cols.remove(category)
    
    heigth = int(np.ceil(len(non_obj_cols)/width))
    
    fig, ax = plt.subplots(heigth, width, figsize=(fig_width, fig_heigt))
    i=0
    j=0
    
    for idx, feature in enumerate(non_obj_cols):
        
        print(idx, feature)
        
        if j >= width:
            j=0
            i+=1
        
        sns.distplot(one[feature], ax=ax[i,j]
                    ,hist=True
                    ,kde=False
                    ,color='red'
                    ,label='attempted'
                    ,hist_kws={"histtype": "step", "linewidth":3})
        sns.distplot(zero[feature], ax=ax[i,j]
                    ,hist=True
                    ,kde=False
                    ,color='green'
                    ,label='not attempted'
                    ,hist_kws={"histtype": "step", "linewidth":3})
        j+=1
    
    plt.show()

hist_class('target', df_imp, 4, 30, 30)


###############################
## Histogram by target variables with random sampling (central limit theorem):
###############################

def hist_class_random_sample(category, df, num_sample, iteration, width, fig_width, fig_heigt):
    
    from joblib import Parallel, delayed
    from randstr import randstr
    
    # Separate the dataset
    one = df[df[category]==1]
    zero = df[df[category]==0]
    
    # random sampling function for class 1
    def sample_one(i):
        samp_one = one.sample(n=num_sample)
        samp_one['key'] = randstr()
        return samp_one
    
    # iterate for many times and put into dataframe
    all_one = Parallel(n_jobs=2, backend="threading", verbose=1)(delayed(sample_one)(i) for i in range(iteration))
    all_one_df = pd.concat(all_one)
    
    # calculate the mean per each sampling key
    one_rand_samp_means = all_one_df.groupby('key').mean()
    
    # random sampling function for class 0
    def sample_zero(i):
        samp_zero = zero.sample(n=num_sample)
        samp_zero['key'] = randstr()
        return samp_zero
    
    # iterate for many times and put into dataframe
    all_zero = Parallel(n_jobs=2, backend="threading", verbose=1)(delayed(sample_zero)(i) for i in range(iteration))
    all_zero_df = pd.concat(all_zero)
    
    # calculate the mean per each sampling key
    zero_rand_samp_means = all_zero_df.groupby('key').mean()
    
    # visualisation
    # calculate the number of rows
    non_obj_cols = [col for col in df.columns if df[col].dtypes != 'O']
    non_obj_cols.remove(category)
    heigth = int(np.ceil(len(non_obj_cols)/width))
    
    # plot individual histogram
    fig,ax = plt.subplots(heigth, width, figsize=(fig_width, fig_heigt))
    i=0
    j=0
    
    for idx, feature in enumerate(non_obj_cols):
        print(idx, feature)
        
        if j>=width:
            j=0
            i+=1
            
        sns.distplot(one_rand_samp_means[feature]
                    ,ax=ax[i,j]
                    ,hist=True
                    ,kde=False
                    ,color='red'
                    ,label='1'
                    ,hist_kws={"histtype": "step", "linewidth":3})
        sns.distplot(zero_rand_samp_means[feature]
                    ,ax=ax[i,j]
                    ,hist=True
                    ,kde=False
                    ,color='green'
                    ,label='0'
                    ,hist_kws={"histtype": "step", "linewidth":3})
        j+=1
    
    plt.show()

hist_class_random_sample('target', df_imp, 200, 5000, 4, 30, 30)

###############################
## Boxplot
###############################

def boxplot(df, width, fig_width, fig_heigt):
    
    non_obj_cols = [col for col in df.columns if df[col].dtypes != 'O']
    
    heigth = int(np.ceil(len(non_obj_cols)/width))
    
    fig, ax = plt.subplots(heigth, width, figsize=(fig_width, fig_heigt))
    i=0
    j=0
    
    for idx, feature in enumerate(non_obj_cols):
        
        print(idx, feature)
        
        if j >= width:
            j=0
            i+=1
        
        sns.boxplot(df[feature]
                    ,ax=ax[i,j]
                    )
        j+=1
    
    plt.show()

boxplot(df_imp, 4, 30, 40)

###############################
## Pairplot by variable
###############################

def pairplot(x, hue, df, width, fig_x, fig_y):
    y=[col for col in df.columns if df[col].dtypes != 'O']
    y.remove(x)
    
    height = int(np.ceil(len(y)/width))
    
    fig, ax = plt.subplots(height, width, figsize=(fig_x, fig_y))
    i=0
    j=0
    
    for idx,feature in enumerate(y):
        print(idx, feature)
        
        if j>= width:
            j=0
            i+=1
        sns.scatterplot(x=x
                       ,y=feature
                       ,ax=ax[i,j]
                       ,hue=hue
                       ,alpha=0.7
                       ,data=df)
        j+=1
    
    plt.show()

pairplot('cont_average_discount_percent', 'target', df_pair, 4, 30, 40)
