from cfg import *
from tqdm import tqdm

class MinMaxScaler:
    def __init__(self):
        self.scale_min_dict = None
        self.scale_max_dict = None

    def fit_transform(self, df, sidx):
        cols = df.columns[sidx:]
        min_values = df[cols].min(axis=1)
        max_values = df[cols].max(axis=1)
        self.scale_min_dict = min_values.to_dict()
        self.scale_max_dict = max_values.to_dict()
        ranges = max_values - min_values
        ranges[ranges==0] = 1
        df[cols] = df[cols].subtract(min_values, axis=0).div(ranges, axis=0)
        return df

    def inverse_transform(self, pred):
        for idx in range(len(pred)):
            pred[idx, :] = pred[idx, :] * (self.scale_max_dict[idx] - self.scale_min_dict[idx]) + self.scale_min_dict[idx]
        return pred

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)

def scaling_category(train_df):
    category = train_df[train_df.columns[:4]]

    min_value = category.min(axis=0)
    max_value = category.max(axis=0)

    train_df[train_df.columns[:4]] = category.subtract(min_value, axis=1).div(max_value - min_value, axis=1)
    return train_df

def make_train_data(data, keyword_df, discount_df, train_size=CFG.train_window_size, predict_size=CFG.predict_size, feature_size = 4):
    # Reference : https://dacon.io/competitions/official/236129/codeshare/8691?page=1&dtype=recent
    num_rows = len(data)
    window_size = train_size + predict_size
    series_size = len(data.iloc[0, feature_size:]) - window_size + 1
    
    input_data = np.empty((num_rows * series_size, train_size, len(data.iloc[0, :feature_size]) + 3)).astype(np.float16)
    target_data = np.empty((num_rows * series_size, predict_size)).astype(np.float16)
    
    for i in tqdm(range(num_rows), desc='make_train_dataset'):
        encode_info = np.array(data.iloc[i, :feature_size])
        sales_data = np.array(data.iloc[i, feature_size:])
        keyword_data = keyword_df[keyword_df['브랜드'] == encode_info[-1]].iloc[0, 1:]
        discount_data = np.array(discount_df.iloc[i])

        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            keyword = keyword_data[j:j+window_size]
            discount = discount_data[j:j+window_size]

            temp_data = np.column_stack(
                (
                    keyword[:train_size], 
                    discount[:train_size], 
                    np.tile(encode_info, (train_size, 1)), 
                    window[:train_size]
                    )
                ).astype(np.float16)
            
            input_data[i * series_size + j] = temp_data
            target_data[i * series_size + j] = window[train_size:]
    
    return input_data, target_data, series_size

def train_val_split(data, input_data, target_data, series_size, train_size=CFG.train_window_size, predict_size=CFG.predict_size):
    # Reference : https://dacon.io/competitions/official/236129/codeshare/8691?page=1&dtype=recent
    val_index = sorted(random.sample(range(series_size),int(series_size*0.2)))
    
    num_rows=len(data)
    
    val_inputs = [] # 일 순서대로 담을 예정
    val_targets = [] 
    for i in tqdm(val_index, desc='split validation set'):
        inputs = np.empty((num_rows, train_size, input_data.shape[2]))
        targets = np.empty((num_rows, predict_size))
        for j in range(num_rows):
            inputs[j] = input_data[j * series_size + i]
            targets[j] = target_data[j * series_size + i]
        val_inputs.append(inputs)
        val_targets.append(targets)
        
    train_series_size = series_size - len(val_index)
    
    train_input = np.empty((num_rows * train_series_size, train_size, input_data.shape[2])).astype(np.float16)
    train_target = np.empty((num_rows * train_series_size, predict_size)).astype(np.float16)
    
    # train 데이터 생성
    k = 0
    for i in tqdm(range(series_size), desc='split train set'):
        if i not in val_index:
            for j in range(num_rows):
                train_input[k] = input_data[j * series_size + i]
                train_target[k] = target_data[j * series_size + i]
                k += 1
                
    return train_input, train_target, val_inputs, val_targets

def make_predict_data(data, keyword_df, discount_df, train_size=CFG.train_window_size, feature_size=4):
    # Reference : https://dacon.io/competitions/official/236129/codeshare/8691?page=1&dtype=recent
    num_rows = len(data)
    
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :feature_size]) + 3)).astype(np.float16)
    
    for i in tqdm(range(num_rows), desc='make_predict_data'):
        encode_info = np.array(data.iloc[i, :feature_size])
        sales_data = np.array(data.iloc[i, -train_size:])
        
        window = sales_data[-train_size : ]
        keyword_data = keyword_df[keyword_df['브랜드'] == encode_info[-1]].iloc[0, -train_size:]
        discount_data = np.array(discount_df.iloc[i, -train_size:])
        
        temp_data = np.column_stack(
            (
                keyword_data[:train_size], 
                discount_data[:train_size], 
                np.tile(encode_info, (train_size, 1)), 
                window[:train_size]
                )
            ).astype(np.float16)
        input_data[i] = temp_data
    
    return input_data