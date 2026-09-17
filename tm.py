import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import am
import csv

# hosted dataset on my github repo
trainData = "https://github.com/joelpriyanth10/ML-Final-Project/raw/refs/heads/main/fashion-mnist_train.csv"
testData = "https://github.com/joelpriyanth10/ML-Final-Project/raw/refs/heads/main/fashion-mnist_test.csv"
train_df = pd.read_csv(trainData)
test_df = pd.read_csv(testData)

# droped the label column
X_train = train_df.drop('label', axis=1)
X_test = test_df.drop('label', axis=1)

# pre processing the data to normalize pixel values between 0 and 1
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# funtion for gaussian noise
def gnoise(X, sigma=0.2):
    noise = np.random.normal(0.0, sigma, X.shape) # getting normal distribution

    return np.clip(X + noise, 0., 1.) # makes sure value not negative or > 1

# function for salt and pepper noise
def spnoise(X, sigma=0.2):
    spvalue = sigma/2
    copy = X.copy()

    #X[rnd < spvalue] = 0 
    #X[rnd > 1 - spvalue] = 1 

    rnd = np.random.rand(*X.shape) # getting a random pixel

    # setting it to either salt or pepper
    copy[rnd < spvalue] = 0 
    copy[rnd > 1 - spvalue] = 1 
    return copy


# function to train the model
def train(X, h1, h2, h3, lr, epochs, type, batch=128):
    hp = am.set_params(784, h1, h2, h3)
    losses = [] # track the loss
    for i in range(epochs):
        rnd = np.random.permutation(len(X)) # pick a random index
        newX = X[rnd] # get the values from X
        error = 0
        iter = 0

        for j in range(0, len(newX), batch):
            imgs = []

            for k in range(j, min(j + batch, len(newX))):
                imgs.append(newX[k])

            arr = np.array(imgs)

            noise = type(arr, 0.2)
            
            hist = am.forward_pass(hp, noise)
            error += am.mse(hist[-1], arr)
            iter += 1
            derivs = am.back_pass(noise, arr, hist, hp)
            # fixed really large gradients
            for d in range(len(derivs)):
                derivs[d] = np.clip(derivs[d], -1.0, 1.0)
            hp = am.update_wts(hp, derivs, lr)
        
        mean = error / iter
        losses.append(mean)
        #print(f"Total epoch: {epochs}")

        # need the mean to compare
        print(f"Epoch {i+1}\tLoss: {mean}")


    return hp, losses

if __name__ == "__main__":
    # changing these hyperparameters 
    epochs = [50, 100]
    lrate = [0.001, 0.01]
    dimensions = [(512, 256, 128), (256, 128, 64)]
    
    results = [] # storing results for the log file
    i = 1

    # running for gaussian noise
    for epoch in epochs:
        for lr in lrate:
            for dim in dimensions:
                h1, h2, h3 = dim
                print(f"\nExperiment #: {i}")
                print(f"hyperparameters: Epochs: {epoch}, LR: {lr}, Hidden Layer dimensions: {h1},{h2},{h3}")

                hp, hist = train(X_train, h1, h2, h3, lr, epoch, gnoise)

                plt.figure(figsize=(8, 5))
                plt.plot(range(1, len(hist) + 1), hist)
                plt.title(f'Learning Curve Exp #{i}: Epochs={epoch}, LR={lr}')
                plt.xlabel('Epochs')
                plt.ylabel('MSE')
                plt.savefig(f'LR{i}.png')
                plt.close()

                # get test error
                noise = gnoise(X_test, 0.2)
                final = am.forward_pass(hp, noise)[-1]
                error = am.mse(final, X_test)

                # final images
                num = X_test[:8]
                noise = gnoise(num, 0.2)
                denoise = am.forward_pass(hp, noise)[-1]

                rows = ["Original:", "Noise:", "Result:"]
                data = [num, noise, denoise]
                fig, axis = plt.subplots(3, 8, figsize=(16, 6))

                for j in range(3):
                    for k in range(8):
                        axis[j, k].imshow(data[j][k].reshape(28, 28), cmap='gray')
                        axis[j, k].axis('off')

                        if k == 0: # puts the labes to the left of column
                            axis[j, k].text(-10, 14, rows[j], va='center', ha='right')
                
                plt.suptitle(f"Gaussian Experminet #{i}: LR={lr}, Epochs={epoch}, Hidden Layer dimensions: {h1}, {h2}, {h3}")
                plt.savefig(f'Gauss{i}.png')
                plt.close()

                # save results
                results.append({"Experiment #": i, "Noise": "Gaussian", "Learning Rate": lr, "Epochs": epoch, "Hidden Layer dimmensions": f"{h1}, {h2}, {h3}", "Train error": round(hist[-1], 5), "Test error": round(error, 4)})
                i += 1

    # salt pepper noise
    for epoch in epochs:
        for lr in lrate:
            for dim in dimensions:
                h1, h2, h3 = dim
                print(f"\nExperiment #: {i}")
                print(f"hyperparameters: Epochs: {epoch}, LR: {lr}, Hidden Layer dimensions: {h1},{h2},{h3}")

                hp, hist = train(X_train, h1, h2, h3, lr, epoch, spnoise)

                plt.figure(figsize=(8, 5))
                plt.plot(range(1, len(hist) + 1), hist)
                plt.title(f'Learning Curve Exp #{i}: Epochs={epoch}, LR={lr}')
                plt.xlabel('Epochs')
                plt.ylabel('MSE')
                plt.savefig(f'LR{i}.png')
                plt.close()

                # get test error
                noise = spnoise(X_test, 0.2)
                final = am.forward_pass(hp, noise)[-1]
                error = am.mse(final, X_test)

                # final images
                num = X_test[:8]
                noise = spnoise(num, 0.2)
                denoise = am.forward_pass(hp, noise)[-1]

                rows = ["Original:", "Noise:", "Result:"]
                data = [num, noise, denoise]
                fig, axis = plt.subplots(3, 8, figsize=(16, 6))

                for j in range(3):
                    for k in range(8):
                        axis[j, k].imshow(data[j][k].reshape(28, 28), cmap='gray')
                        axis[j, k].axis('off')

                        if k == 0:
                            axis[j, k].text(-10, 14, rows[j], va='center', ha='right')
                
                plt.suptitle(f"Salt and Pepper Experiment #{i}: LR={lr}, Epochs={epoch}, Hidden Layer dimensions: {h1}, {h2}, {h3}")
                plt.savefig(f'SP{i}.png')
                plt.close()

                # save results
                results.append({"Experiment #": i, "Noise": "Salt and Pepper", "Learning Rate": lr, "Epochs": epoch, "Hidden Layer dimmensions": f"{h1}, {h2}, {h3}", "Train error": round(hist[-1], 5), "Test error": round(error, 4)})
                i += 1

    # make log file
    with open('results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    gauss = results_df[results_df["Noise"] == "Gaussian"]
    gmean = gauss.groupby("Learning Rate")["Train error"].mean()
    plt.plot(gmean.index.astype(str), gmean.values, label='Gaussian')
    sp = results_df[results_df["Noise"] == "Salt and Pepper"]
    spmean = sp.groupby("Learning Rate")["Train error"].mean()
    plt.plot(spmean.index.astype(str), spmean.values, label='Salt and Pepper')
    plt.title('MSE vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MSEplot.png')


