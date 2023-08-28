import gc
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cfg import *
from make_dataset import *
from model import Model

def PSFA(pred, target, indexs_cat):
    # Reference : https://dacon.io/competitions/official/236129/codeshare/8662?page=1&dtype=recent
    total_psfa, max_psfa, min_psfa = 0, 0, 1
    for i in range(pred.shape[0]):
        psfa = 1
        for cat in indexs_cat.keys():
            ids = indexs_cat[cat]
            for day in range(21):
                total_sell = np.sum(target[i][ids, day])
                pred_values = pred[i][ids, day]
                target_values = target[i][ids, day]
                
                denominator = np.maximum(target_values, pred_values)
                diffs = np.where(denominator != 0, np.abs(target_values - pred_values) / denominator, 0)
                
                if total_sell != 0:
                    sell_weights = target_values / total_sell
                else:
                    sell_weights = np.ones_like(target_values) / len(ids)
                
                if not np.isnan(diffs).any():
                    psfa -= np.sum(diffs * sell_weights) / (21 * 5) 
        
        total_psfa += psfa
        max_psfa = max(max_psfa, psfa)
        min_psfa = min(min_psfa, psfa)
    
    return total_psfa / pred.shape[0], max_psfa, min_psfa

def validation(model, val_input, val_target, scaler, indexs_cat, device):
    # Reference : https://dacon.io/competitions/official/236129/codeshare/8662?page=1&dtype=recent
    pred = np.empty((len(val_input), len(val_input[0]), CFG.predict_size))
    target = np.empty((len(val_input), len(val_input[0]), CFG.predict_size))
    
    with torch.no_grad():
        for i in trange(len(val_input), desc='validation'):
            val_dataset = CustomDataset(val_input[i], val_target[i])
            val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
            
            pred_ = []
            target_ = []
            
            for X, Y in iter(val_loader):
                X = X.to(device)
                output = model(X).cpu().numpy()
                
                pred_.extend(output)
                target_.extend(Y.numpy())
            
            target[i] = np.array(target_)
            pred[i] = np.array(pred_)
    
    # pred, target shape (68, 18590, 21)
    # inverse transform
    for i in range(len(pred)):
        pred[i] = scaler.inverse_transform(pred[i])
        target[i] = scaler.inverse_transform(target[i])

    pred = np.round(pred, 0).astype(int)
    target = np.round(target, 0).astype(int)
    
    return PSFA(pred, target, indexs_cat)

def train(train_loader, val_input, val_target, model, optimizer, scaler, indexs_cat, device):
    best = 0
    best_model = model
    
    def criterion(pred, true):
        # RMSE
        mse = F.mse_loss(pred, true)
        return mse ** 0.5

    for epoch in range(CFG.epochs):
        total_loss = 0.
        model.train()
        for x, y in tqdm(train_loader, desc='training'):
            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).float()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

        psfa, max_psfa, min_psfa = validation(model, val_input, val_target, scaler, indexs_cat, device)

        print(f'Epoch = {epoch+1} | loss = {total_loss/len(train_loader):5f} | val_mean_psfa = {psfa:.5f} | val_max_psfa = {max_psfa:.5f} | val_min_psfa = {min_psfa:.5f}')

        if best < psfa:
            best = psfa
            best_model = model
    
    torch.save(best_model.state_dict(), f'artifact/{best:.5f}-{min_psfa:.5f}.pt')
    print(f'best mean psfa = {best:.5f}')

if __name__ == "__main__":
    assert torch.cuda.is_available() == True, 'Not working GPU'
    
    seed_everything(CFG.seed)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device = {device}')

    train_df = pd.read_csv('data/scaled_train.csv')
    discount_df = pd.read_csv('data/scaled_discount.csv')
    keyword_df = pd.read_csv('data/scaled_keyword.csv')

    indexs_cat={}
    for bigcat in train_df['대분류'].unique():
        indexs_cat[bigcat] = list(train_df.loc[train_df['대분류']==bigcat].index)

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df, 4)
    
    # make dataset, split
    input_data, target_data, series_size = make_train_data(scaled_train, keyword_df, discount_df)
    
    """
    input_data = (5402600, 100, 7), target_data = (5402600, 21), series_size = 340
    train_input, train_target shape = (4322080, 100, 7), (4322080, 21)
    val_input, val_target shape = (68, 15890, 100, 7), (68, 15980, 21)
    """
    train_input, train_target, val_input, val_target = train_val_split(train_df, input_data, target_data, series_size)
    
    # free memory
    del input_data, target_data, series_size
    gc.collect()
    
    train_dataset = CustomDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size = CFG.batch_size, shuffle=True, num_workers=0)

    # load model, cost function, optimizer
    model = Model(hidden_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    
    train(train_loader, val_input, val_target, model, optimizer, scaler, indexs_cat, device)
