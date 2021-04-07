import tensorflow as tf 
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



class Trader:
    def __init__(self):

        self.stock_num = 0
        self.p = 0
        self.predict_day = 0
        self.test_data = []

    def build(self,data):
        
        model = tf.keras.Sequential([
        
        tf.keras.layers.LSTM(500,input_shape=data.shape[-2:]),
        tf.keras.layers.Dense(2)])

        
        model.compile(optimizer='sgd', loss='mean_squared_error')
        return model

    def train(self,data):
        
        self.data = data
        print("資料前處理")
        x,y= self.data_pre_new(data)
        print(x)
        print(y)
        x,y =self.unison_shuffled_copies(x,y)
        x,y,test_x,test_y = self.cut(x,y)
        model =self.build(x)
        model.fit(x,y,validation_data = [test_x,test_y] ,epochs=500)
        model.save('path_to_saved_model', save_format='tf')
        print("模型儲存完畢")


    def data_pre_new (self,data):
        h = len(data)
        target = []
        for i in range(0,h):
            target.append(data[i][0]/200)
        for i in range(0,h):
            for j in range(0,len(data[i])):
                data[i][j] = data[i][j]/200
        x = []
        y = []
        count = 0
        
        for i in range(0,h):
            if i + 5 > h-1 :
                break
            batch = []
            #前3天
            for j in range(0,3):
                a = data[i+j]
                batch.append(a)

            x.append(batch)
            ans = []
            #第3到4天
            
            temp = 0
            for k in range(3,5):
                temp = target[i+k]
                ans.append(target[i+k])
            
            
            y.append(ans)
            x_n = np.array(x)
            y_n = np.array(y)
        

        return x_n,y_n



    def predict_data_pre(self):
        
        batch = []
        for i in range(self.predict_day-3,self.predict_day):
            new_data = []
            for j in range(0,len(self.test_data[i])):
                new_data.append(self.test_data[i][j]/200)
            batch.append(new_data)
        x = [batch]
        print("當天",x )
        return np.array(x) 

    def cut(self,x_n,y_n):
        h = x_n.shape[0]
        spli = int(0.9*h)
        tran_x = x_n[:spli]
        test_x =  x_n[spli:]
        tran_y = y_n[:spli]
        test_y =  y_n[spli:]

        return tran_x,tran_y,test_x,test_y


    def unison_shuffled_copies(self,a, b):
        
        np.random.seed(0)
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def predict_action(self,data):
        self.predict_day += 1
        if self.predict_day < 3:
            self.test_data.append(data)
            return 0
        else:
            self.test_data.append(data)
            x = self.predict_data_pre()
            new_model = tf.keras.models.load_model('path_to_saved_model')
            predict = new_model.predict(x)
            predict *= 200
            print(predict)
            action = 0

            if predict[0][1]-predict[0][0] > 0:
                
                if self.stock_num == 0:
                    action = 1
                    self.stock_num = 1
                elif self.stock_num == 1:
                    action = 0
                elif self.stock_num == -1:
                    action = 1
                    self.stock_num = 0

            elif predict[0][1]-predict[0][0] < 0:
                if self.stock_num == 0:
                    action = -1
                    self.stock_num =  -1
                elif self.stock_num == 1:
                    action = -1
                    self.stock_num = 0
                elif self.stock_num == -1:
                    action = 0
                    
            return action
        



def rmse(predictions, targets): 
    return np.sqrt(((predictions - targets) ** 2).mean()) 


def load_data(path):
    ans = []
    df = pd.read_csv(path,header = None,encoding='utf-8')
    h = df.shape[0]
    for i in range(0,h):
        row = []
        for key in df:
            
            row.append((df[key][i]))#將數字等比例縮小
        ans.append(row)
    return ans

def count(test_d,ans):
    total = 0
    print(test_d)
    num = 0
    money = 300
    for i in range(1,20):
        
        num +=ans[i-1]
        money += test_d[i][0]*ans[i-1]*-1
        print(i,num,money)
    if num > 0:
        money +=  test_d[19][3]
    elif num <0 :
        money -=  test_d[19][3]

    return money-300





if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    trader = Trader()
    trader.train(training_data)
    
    testing_data = load_data(args.testing)
    with open(args.output, 'w') as output_file:
        count = 1
        for row in testing_data:
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row)
            if count < len(testing_data):
                output_file.write(str(action)+"\n")
            count +=1


