import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import csv
import os
import warnings
warnings.filterwarnings("ignore")

# Function to compare best parameters for SVR and SVR-W models
def Compare_bestPara(X_train_Standard, X_test_Standard, y_train_Standard, y_train, y_test, scaler_minmax, model, filePath, province):
    if model == 'SVR':
        # Define parameter grid for SVR
        param_grid = {
            'C': [0.001, 0.01, 0.1],  # Regularization parameter
            'gamma': [0.1, 0.01],  # Kernel parameter
            'kernel': ['rbf']  # Kernel type
        }
        regressor = svm.SVR()
        # Create GridSearchCV instance
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=10)
        grid_search.fit(X_train_Standard, np.ravel(y_train_Standard))
        # Best parameters and best score
        print(f"Best parameters: {grid_search.best_params_}")
        best_model = svm.SVR(**grid_search.best_params_)
        best_model.fit(X_train_Standard, y_train_Standard)
        y_pred1 = best_model.predict(X_test_Standard)
        y_pred1 = scaler_minmax.inverse_transform(y_pred1.reshape(-1, 1))

    if model == 'SVR-W':
        # Define parameter grid for SVR-W
        param_grid = {
            'C': [0.001, 0.01, 0.1],  # Regularization parameter
            'gamma': [0.1, 0.01],  # Kernel parameter
            'kernel': ['rbf'],  # Kernel type
            't': [10, 15, 20]  # Custom parameter t
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_score = 100000
        best_params = None

        # Iterate through all parameter combinations
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                for t in param_grid['t']:
                    fold_scores = []

                    # 5-fold cross-validation
                    for train_index, test_index in kf.split(X_train_Standard):
                        X_train_fold, X_test_fold = X_train_Standard[train_index], X_train_Standard[test_index]
                        y_train_fold, y_test_fold = y_train_Standard[train_index], y_train_Standard[test_index]

                        svr_model = svm.SVR(C=C, gamma=gamma, kernel='rbf')
                        weightList = GetWeight2(y_train[train_index], t)

                        svr_model.fit(X_train_fold, y_train_fold, sample_weight=weightList)
                        y_pre_fold = svr_model.predict(X_train_fold)
                        mse = mean_squared_error(y_train_fold, y_pre_fold)
                        fold_scores.append(mse)

                    avg_rmse = np.mean(fold_scores)

                    # Compare current parameter set's average RMSE with the best score
                    if avg_rmse < best_score:
                        best_score = avg_rmse
                        best_params = {'C': C, 'gamma': gamma, 't': t}
                        print(f"Best parameters: {best_params}")
                        print(f"Best RMSE: {abs(avg_rmse)}")

        svr_model = svm.SVR(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
        weightList = GetWeight2(y_train, best_params['t'])
        svr_model.fit(X_train_Standard, y_train_Standard, sample_weight=weightList)
        y_pred1 = svr_model.predict(X_test_Standard)
        y_pred1 = scaler_minmax.inverse_transform(y_pred1.reshape(-1, 1))

        # Save best parameters to CSV
        filep = filePath + '_best_params.csv'
        with open(filep, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['C', 'gamma', 't', 'best_RMSE'])
            writer.writeheader()
            writer.writerow({'C': best_params['C'], 'gamma': best_params['gamma'], 't': best_params['t'],
                             'best_RMSE': abs(best_score)})

    return y_pred1

# Function to get and clean data
def get_data(train_filePath, test_filePath):
    columns_to_read = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3_8h', 'e', 'sp', 'ssrd', 't2m',
                       'tp', 'u10', 'v10', 'RH', 'blh', 'tcc', 'lon', 'lat', 'tropospheric_HCHO_column_number_density']

    train = pd.read_csv(train_filePath, usecols=columns_to_read)
    train = clean_data(train)
    y_train = train['O3_8h']
    X_train = train.iloc[:, [*range(0, 5), *range(6, 19)]]

    test = pd.read_csv(test_filePath, usecols=columns_to_read)
    test = clean_data(test)
    y_test = test['O3_8h']
    X_test = test.iloc[:, [*range(0, 5), *range(6, 19)]]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Standardize the data
    scaler_zscore = StandardScaler()
    X_train_Standard = scaler_zscore.fit_transform(X_train.values)
    X_test_Standard = scaler_zscore.transform(X_test.values)
    y_train_Standard = scaler_zscore.fit_transform(y_train.reshape(-1, 1))
    y_test_Standard = scaler_zscore.transform(y_test.reshape(-1, 1))

    print("The number of training data:", len(y_train_Standard))
    print("The number of test data:", len(y_test))
    return X_train_Standard, X_test_Standard, y_train_Standard, y_train, y_test, scaler_zscore

# Function to get weight for SVR-W model
def GetWeight2(y_train, t):
    n = float(t)
    y_train = np.array(y_train)
    sorted_array = np.sort(y_train)
    percentile = len(sorted_array) // 100
    percentile1 = len(sorted_array) // 10
    small_para = sorted_array[percentile]
    large_para = sorted_array[-percentile1]
    WeightList = np.ones(len(y_train))
    ind1 = np.where(y_train >= large_para)
    ind2 = np.where(y_train <= small_para)
    WeightList[ind1[0]] = (y_train[ind1[0]] / large_para) ** n
    WeightList[ind2[0]] = (small_para / y_train[ind2[0]]) ** n
    print("p1 and p2:", small_para, large_para)
    return WeightList

# Function to clean data (you need to implement this based on your data)
def clean_data(data):
    # Implement any cleaning steps needed, such as handling missing values, outliers, etc.
    return data.dropna()

# Function to calculate slope and RMSE
def getSLOPERMSER(y_test, y_pred1):
    # Calculate Slope and RMSE y_test is true label
    y_test = y_test.flatten()
    y_pred1 = y_pred1.flatten()
    slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred1)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred1))
    return slope, RMSE

# Function to calculate MSE balance
def getMSE_balance(Y_Pre, Y):
    #Y为 true label

    interval = 20
    item = 10 #循环次数
    min_day = 5

    #######################根据真实数据，挑选预测数据，得到平衡的数据##################################
    Y_Pre_pd = pd.DataFrame(Y_Pre)
    Y_pd = pd.DataFrame(Y)
    slope_Ava = 0; slope_std = 0; slope_list = []
    RMSE_Ava = 0; RMSE_std = 0; RMSE_list = []

    for j in range(item):
        # 计算最大值和最小值
        min_value = np.min(Y) - 0.001
        max_value = np.max(Y) + 0.001  # 使得最大最小值能被包含进去
        interval_width = (max_value - min_value) / interval
        # 定义区间
        intervals = [(min_value + i * interval_width, min_value + (i + 1) * interval_width) for i in range(interval)]

        # 先确定每个范围的数量
        num_inter_list = np.ones(interval)
        # 确定数量
        for i, (start, end) in enumerate(intervals):
            new_Y = Y_pd[(Y_pd[0] >= start) & (Y_pd[0] < end)]
            num_inter_list[i] = len(new_Y)
        num_inter_list = np.vstack(num_inter_list)
        # Remove elements where num_inter_list is 0 and the corresponding intervals
        # Convert intervals and num_inter_list to lists for easier manipulation
        intervals_list = list(intervals)
        num_inter_list_cleaned = []
        intervals_cleaned = []

        for i, count in enumerate(num_inter_list):
            if count > min_day:  # Keep only non-zero elements
                num_inter_list_cleaned.append(count)
                intervals_cleaned.append(intervals_list[i])
        intervals = intervals_cleaned
        num_inter_list = num_inter_list_cleaned
        #取区间内最小值的一半
        min_num = max(round_up(np.min(num_inter_list) / 2),1)

        #开始每个区间随机选
        y_list = []
        Y_Pre_list = []
        for i, (start, end) in enumerate(intervals):
            new_Y = Y_pd[(Y_pd[0] >= start) & (Y_pd[0] < end)]
            # 从数组中随机抽取min_num个数
            new_Y = new_Y.sample(min_num,random_state=j)  #随机种子，这样能复现
            new_Y_Pre = Y_Pre_pd.loc[new_Y.index]
            y_list.append(np.array(new_Y))
            Y_Pre_list.append(np.array(new_Y_Pre))
        Y_balance = np.array(y_list).flatten()
        y_pre_balance = np.array(Y_Pre_list).flatten()
        ######################计算平衡数据状态下各项指标##############################
        slope_B, RMSE_B = getSLOPERMSER(Y_balance, y_pre_balance)
        slope_list.append(slope_B)
        RMSE_list.append(RMSE_B)

    slope_Ava, slope_std = getMeanStd(slope_list)
    RMSE_Ava,RMSE_std = getMeanStd(RMSE_list)

    return slope_Ava, slope_std,  RMSE_Ava, RMSE_std

def getMeanStd(X):
    X = np.array(X)
    mean_d = np.mean(X) if X.size > 0 else float('nan')
    if X.size <= 0:
        print(X)
    # mean_d = np.mean(X)
    std_d = np.std(X, ddof=0)
    return mean_d,std_d

def round_up(number):
    a = int(number) + (1 if number > int(number) else 0)
    return a


# Main function to train models for all provinces
def train_all_province(modelList,filePath,save_path):
    train_files = []
    test_files = []
    for file in os.listdir(filePath):
        if file.endswith("train data.csv"):
            train_files.append(os.path.join(filePath, file))
        elif file.endswith("test data.csv"):
            test_files.append(os.path.join(filePath, file))

    # Sort files
    sorted(train_files, key=lambda x: x.lower())
    sorted(test_files, key=lambda x: x.lower())
    name_list = ['Qinghai', 'Shanghai', 'Xizhang']

    # Loop over each province
    for i, train_file in enumerate(train_files):
        X_train_Standard, X_test_Standard, y_train_Standard, y_train, y_test, scaler_zscore = get_data(
            train_file, test_files[i])
        print(test_files[i])
        print(name_list[i])

        for model in modelList:
            print(f"Training {model} model for {name_list[i]}...")
            y_pred1 = Compare_bestPara(X_train_Standard, X_test_Standard, y_train_Standard, y_train, y_test,
                                       scaler_zscore, model, save_path, name_list[i])

            slope_B, slope_B_std, RMSE_B, RMSE_B_std = getMSE_balance(y_pred1, y_test)

            # Store results
            save_results(name_list[i], model, slope_B, slope_B_std, RMSE_B, RMSE_B_std,save_path)

# Function to save results
def save_results(province, model, slope_B, slope_B_std, RMSE_B, RMSE_B_std,save_path):
    result_list = [
        [province, model,  slope_B, slope_B_std, RMSE_B, RMSE_B_std]
    ]
    result_df = pd.DataFrame(result_list, columns=[
        'Province', 'Model', 'Slope_B', 'Slope_B_std', 'RMSE_B', 'RMSE_B_std'])
    save_path = os.path.join(save_path, "model_results.csv")
    result_df.to_csv(save_path, mode='a', header=False, index=False)
    print("Results saved successfully!")

if __name__ == '__main__':
    modelList = ['SVR-W']
    # Load training data and test data
    filePath = ""
    # save result
    savePath = ""
    train_all_province(modelList,filePath,savePath)

