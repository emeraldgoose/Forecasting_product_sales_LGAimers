from tqdm import tqdm
from torch.utils.data import DataLoader
from cfg import *
from make_dataset import *
from model import Model

def inference(test_data, model, scaler, device, artifact_name):
    model.load_state_dict(torch.load(f'artifact/{artifact_name}.pt'))
    testset = CustomDataset(test_data)
    test_loader = DataLoader(testset, batch_size=10, shuffle=False)
    
    prediction =[]
    with torch.no_grad():
        for X in tqdm(iter(test_loader), desc='Inference'):
            X = X.to(device).float()
            output = model(X)
            output = output.cpu().numpy()
            prediction.extend(output)

    prediction = scaler.inverse_transform(np.array(prediction))
    prediction = np.round(prediction, 0).astype(int)
    
    if np.sum(prediction < 0):
        prediction = (prediction > 0) * prediction
    
    submission = pd.read_csv('data/sample_submission.csv')
    submission.iloc[:, 1:] = prediction
    
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    assert torch.cuda.is_available() == True, 'Not working GPU'
    
    seed_everything(CFG.seed)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device = {device}')

    train_df = pd.read_csv('data/train_preprocessed.csv')
    discount_df = pd.read_csv('data/discount.csv').fillna(0)
    keyword_df = pd.read_csv('data/brand_keyword_cnt_preprocessed.csv').fillna(0)

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df)
    
    model = Model(hidden_size=128).to(device)
    
    test_data = make_predict_data(scaled_train, keyword_df, discount_df)
    
    inference(test_data, model, scaler, device, '0.60364-0.55121')
