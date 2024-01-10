
from src.colorDetection import knn_handler
from src.colorDetection import dataCleaning
from pathlib import Path


train_data = './src/data/Data.csv'
test_data = './src/data/colors.csv'
model_path = Path('../../models/')


    




if __name__ == '__main__':
    #data Preprocessing
    X, y = dataCleaning.loadDataset(train_data)
    X_vector = dataCleaning.rgb_to_embedding(X)

    #model training
    model = knn_handler.train_model(X_vector, y)
    knn_handler.save_model(model, model_path)