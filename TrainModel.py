import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AutoencoderModel import set_params, forward_pass, mse, back_pass, update_wts
import csv

# loading datasets fron github
test_df = pd.read_csv("https://github.com/joelpriyanth10/ML-Final-Project/raw/refs/heads/main/fashion-mnist_test.csv")
train_df = pd.read_csv("https://github.com/joelpriyanth10/ML-Final-Project/raw/refs/heads/main/fashion-mnist_train.csv")

# pre processing data
X_train = train_df.drop('label', axis=1).values / 255.0
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values / 255.0
y_test = test_df['label'].values

# adding gaussian noise
def gaussian_noise(X, sigma=0.1):
  noise = np.random.normal(0.0, sigma, X.shape)
  return np.clip((X + noise), 0., 1.)

# adding salt and peper noise
def salt_pepper_noise(X, sigma=0.05):
  X_noise = np.copy(X)

  # salt noise
  num_salt = np.ceil(sigma*X.size*0.5)
  location = []
  for i in X.shape:
    random = np.random.randint(0, i-1, int(num_salt))
    location.append(random)
  X_noise[tuple(location)] = 1.0

  # pepper noise
  num_pepper = np.ceil(sigma*X.size*0.5)
  location = []
  for i in X.shape:
    random = np.random.randint(0, i-1, int(num_pepper))
    location.append(random)
  X_noise[tuple(location)] = 0.0

  return X_noise

def train(X_n, X_c, hid_dim, lr, epochs, batch=256):
    # intialize all weights and biases
    W1, b1, W2, b2 = set_params(784, hid_dim)
    history = []

    for e in range(epochs):
        # shuffle the batches
        s = np.random.permutation(len(X_n))
        X_n = X_n[s]
        X_c = X_c[s]

        loss = 0
        runs = 0

        for i in range(0, len(X_n), batch):
            Xn_b = X_n[i : i + batch]    # (256, 784)
            Xc_b = X_c[i : i + batch]

            # do a forward pass with the noisy data
            Z1, A1, Z2, A2 = forward_pass(W1, b1, W2, b2, Xn_b)

            # find the loss
            loss += mse(A2, Xc_b)
            runs += 1

            # do a backward pass
            dW1, db1, dW2, db2 = back_pass(Xn_b, Xc_b, Z1, A1, A2, W2)

            # update the weights and biases
            W1, b1, W2, b2 = update_wts(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        
        history.append(loss / runs)
        print(f"Epoch {e+1:>3}/{epochs}  loss: {loss / runs:.4f}")
    
    return W1, b1, W2, b2, history

if __name__ == "__main__":
    X_tr_ga = gaussian_noise(X_train)
    X_tr_sp = salt_pepper_noise(X_train)
    X_te_ga = gaussian_noise(X_test)
    X_te_sp = salt_pepper_noise(X_test)
    
    experiments = [
        {"hid_dim": 64,  "lr": 0.01,  "epochs": 50},
        {"hid_dim": 128, "lr": 0.01,  "epochs": 50},
        {"hid_dim": 128, "lr": 0.01, "epochs": 50},
        {"hid_dim": 128, "lr": 0.1,  "epochs": 100},
        {"hid_dim": 256, "lr": 0.01,  "epochs": 50},
    ]

    output = []

    for i, hp in enumerate(experiments):
       hid_dim = hp["hid_dim"]
       lr = hp["lr"]
       epochs = hp["epochs"]
       
       print(f"Experiment {i+1}: hid_dim={hid_dim}, lr={lr}, epochs={epochs}")

       # train guassian noise
       W1, b1, W2, b2, history_g = train(X_tr_ga, X_train, hid_dim, lr, epochs)
       tr_loss_ga = history_g[-1]
       
       # try on test set
       _, _, _, A2_test_ga = forward_pass(W1, b1, W2, b2, X_te_ga)
       te_loss_ga = mse(A2_test_ga, X_test)
       
       # train salt + pepper noise
       W1s, b1s, W2s, b2s, history_s = train(X_tr_sp, X_train, hid_dim, lr, epochs)
       tr_loss_sp = history_s[-1]
       
       # try on test set
       _, _, _, A2_test_sp = forward_pass(W1s, b1, W2, b2, X_te_sp)
       te_loss_sp = mse(A2_test_sp, X_test)

       # ── save loss curve plot ──────────────────────────────
       plt.figure(figsize=(7, 3))
       plt.plot(history_g, label='Gaussian')
       plt.plot(history_s, label='Salt & Pepper')
       plt.xlabel('Epoch')
       plt.ylabel('MSE Loss')
       plt.title(f'Exp {i+1} — hid={hid_dim}, lr={lr}, epochs={epochs}')
       plt.legend()
       plt.tight_layout()
       plt.savefig(f'exp{i+1}_loss.png', dpi=120)
       plt.close()

        # ── save reconstruction plot ──────────────────────────
       n = 6
       _, _, _, recon_g = forward_pass(W1,  b1,  W2,  b2,  X_te_ga[:n])
       _, _, _, recon_s = forward_pass(W1s, b1s, W2s, b2s, X_te_sp[:n])

       fig, axes = plt.subplots(4, n, figsize=(12, 7))
       row_labels = ['Clean', 'Gauss noisy', 'Gauss recon', 'S&P recon']
       rows = [X_test[:n], X_te_ga[:n], recon_g, recon_s]
       for r, (label, row_data) in enumerate(zip(row_labels, rows)):
           for c in range(n):
               axes[r, c].imshow(row_data[c].reshape(28, 28), cmap='gray')
               axes[r, c].axis('off')
           axes[r, 0].set_ylabel(label, fontsize=8)
       plt.suptitle(f'Exp {i+1} Reconstructions', y=1.01)
       plt.tight_layout()
       plt.savefig(f'exp{i+1}_recon.png', dpi=120)
       plt.close()
       
       # store results
       output.append({
            "Experiment": i+1,
            "Hidden dim": hid_dim,
            "Learning rate": lr,
            "Epochs": epochs,
            "Noise type": "Gaussian",
            "Train MSE": round(tr_loss_ga, 4),
            "Test MSE":  round(te_loss_ga,  4)})
       output.append({
          "Experiment": i+1,
            "Hidden dim": hid_dim,
            "Learning rate": lr,
            "Epochs": epochs,
            "Noise type": "Salt & Pepper",
            "Train MSE": round(tr_loss_sp, 4),
            "Test MSE":  round(te_loss_sp,  4)
       })

       print(f"  Gaussian train MSE: {tr_loss_ga:.4f},  test MSE: {te_loss_ga:.4f}")
       print(f"  Salt&Pepper train MSE: {tr_loss_sp:.4f},  test MSE: {te_loss_sp:.4f}")

       keys = ["Experiment", "Hidden dim", "Learning rate", "Epochs",
            "Noise type", "Train MSE", "Test MSE"]

    with open('experiment_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output)

    print("\nAll done! Results saved to experiment_results.csv")
    print("Loss curves saved as exp1_loss.png, exp2_loss.png ...")
    print("Reconstructions saved as exp1_recon.png, exp2_recon.png ...")
       

