import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_training_data(n):
    X_training = np.random.normal(loc=2, scale=0.5, size=n)
    Y_training = 2 * X_training
    return X_training, Y_training

def generate_testing_data(s):
    X_testing = np.random.normal(loc=2, scale=0.5, size=s)
    Y_testing = 2 * X_testing
    return X_testing, Y_testing

def main():
    model = LinearRegression()

    test_arr = generate_testing_data(100)
    x_test = test_arr[0].reshape(-1, 1)
    y_test = test_arr[1]
    
    bias = []
    variance = []

    for i in range(1, 11):

        train_arr = generate_training_data(100 * i)
        x_train = train_arr[0].reshape(-1, 1)
        y_train = train_arr[1]
        
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        
        bias.append(np.mean(y_pred - y_test))
        variance.append(np.mean((y_pred - y_test) ** 2))
        
        print(f"Bias in training dataset {i} is {bias[i - 1]}")
        print(f"Variance in training dataset {i} is {variance[i - 1]}")
        print("\n")

    x_values = [i * 100 for i in range(1, 11)]
    
    plt.subplot(2, 1, 1)
    plt.plot(x_values, bias, marker='o', linestyle='-', color='b', label='Bias vs N')
    plt.xlabel('Training sample data size')
    plt.ylabel('Bias')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(x_values, variance, marker='o', linestyle='-', color='b', label='Variance vs N')
    plt.xlabel('Training sample data size')
    plt.ylabel('Variance')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
