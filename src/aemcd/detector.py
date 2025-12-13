import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt

# Anomaly Detector (AE + Robust Gaussian (MCD) + Shewhart/EWMA)
class AE_MCD_Detector:
    def __init__(self, autoencoder, lr=1e-3, batch_size=128, epochs=60,
                 q=0.997, ewma_lambda=0.2, device=None, use_re = True):
        self.ae = autoencoder
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.q = q
        self.ewma_lambda = ewma_lambda
        self.use_re = use_re
        self.mcd = None
        self.ucl_shewhart = None
        self.ucl_ewma = None
        self.ucl_re = None
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ae.to(self.device)
        self.ewma_init = None

        
    # EWMA calculation
    def _ewma(self , x, lam=0.2 , init_value=None):
        z = np.zeros_like(x, dtype=float) 
        if init_value is not None:
            z[0] = lam * x[0] + (1 - lam) * init_value  
        else:

            z[0] = x[0]
        for t in range(1, len(x)):
            z[t] = lam * x[t] + (1 - lam) * z[t-1] 
        return z
    # Training function         
    def fit(self, X_train, val_split = 0.1 ,patience=10, verbose=True):
        # (1) Train AE on NORMAL train data
        X_train = np.asarray(X_train, dtype=np.float32)

        # validation split
        n_val = int(len(X_train) * val_split)
        if n_val <= 0:
            X_train_fit = X_train
            X_val = None
        else:
            X_val = X_train[-n_val:]          # validation set from the end
            X_train_fit = X_train[:-n_val]    # training set from the beginning
        ds = TensorDataset(torch.from_numpy(X_train_fit))  # pytorch dataset 
        g = torch.Generator()
        g.manual_seed(42)                           
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)  # AE training dataloader

        opt = optim.AdamW(self.ae.parameters(), lr=self.lr, weight_decay=1e-4) # optimizer
        loss_fn = nn.MSELoss()                
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.train_history = {'train_loss': [], 'val_loss': [], 'epochs': []}

        # training loop
        for ep in range(1, self.epochs + 1):
            self.ae.train()
            
            losses = []
            # training every epoch
            for (xb,) in dl:
                xb = xb.to(self.device)  #xb is batch window vector（shape: [B, input_dim]）
                xhat, _ = self.ae(xb)    # reconstructed xhat
                loss = loss_fn(xhat, xb) 

                opt.zero_grad()
                loss.backward()       #  backprop
                opt.step()
                losses.append(loss.item())
            avg_train_loss = float(np.mean(losses))
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['epochs'].append(ep)


            
            # validation 
            if X_val is not None:
        
                self.ae.eval()  # turn off dropout batchnorm
                with torch.no_grad():
                    xval_tensor = torch.from_numpy(X_val).to(self.device)
                    xval_hat, _ = self.ae(xval_tensor)
                    val_loss = loss_fn(xval_hat, xval_tensor).item()
        
                self.train_history['val_loss'].append(val_loss)
        
                if verbose:
                        print(f"Epoch {ep:03d}/{self.epochs} | "
                        f"train_loss={avg_train_loss:.6f} | "
                        f"val_loss={val_loss:.6f}")
        
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() 
                                for k, v in self.ae.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {ep}")
                        self.ae.load_state_dict(best_state)
                        break
            else:
                #  no validation set
                if verbose:
                    print(f"Epoch {ep:03d}/{self.epochs} | loss={avg_train_loss:.6f}")
        if best_state is not None:
            self.ae.load_state_dict(best_state)    
        # (2) Extract latent Z_train on all train data 
        self.ae.eval()
        with torch.no_grad():
            X_full_t = torch.from_numpy(X_train_fit).to(self.device) 
            X_recon, Z_tr = self.ae(X_full_t)
            Z_tr = Z_tr.cpu().numpy()
            re_tr = torch.mean((X_full_t - X_recon) ** 2, dim=1).cpu().numpy()
            re_tr_t = np.log1p(re_tr) 

        #  (3) Fit MCD on latent (robust mean/covariance)
        self.mcd = MinCovDet(random_state=42).fit(Z_tr)

        #  (4) Distances on train (NOTE: squared distances)
        d2_tr = self.mcd.mahalanobis(Z_tr)            # squared  Mahalanobis distances
        d_tr = np.sqrt(np.maximum(d2_tr, 0.0))        # for charting (for Shewart)

        # (5) Set UCLs to match ARL0 by empirical quantile approximate ARL0 
        # Shewhart UCL
        self.ucl_shewhart = float(np.quantile(d_tr, self.q))
        # EWMA initial value and UCL
        self.ewma_init = float(np.mean(d_tr))
        ew_tr = self._ewma(d_tr, self.ewma_lambda, init_value=self.ewma_init)
        self.ucl_ewma = float(np.quantile(ew_tr, self.q))
        self.ewma_init = ew_tr[-1]
        # Reconstruction error UCL
        if self.use_re:
            self.ucl_re = float(np.quantile(np.log1p(re_tr), self.q))
        # print UCLs
        if verbose:
            print(f"\n{'='*60}")
            print(f"UCL set done:")
            print(f"  UCL_Shewhart (MD): {self.ucl_shewhart:.4f}")
            print(f"  UCL_EWMA (MD):     {self.ucl_ewma:.4f}")
            if self.use_re:
                print(f"  UCL_RE:            {self.ucl_re:.6f}")
            print(f"{'='*60}\n")    
        

        return self
    def plot_training_history(self):
        """ Plot valid an  train Loss charts """
        if not self.train_history['train_loss']:
            print("No value ！")
            return
        
        plt.figure(figsize=(10, 6))
        epochs = self.train_history['epochs']
        
        plt.plot(epochs, self.train_history['train_loss'], 
                label='Training Loss', linewidth=2, marker='o', markersize=3)
        
        if self.train_history['val_loss']:
            plt.plot(epochs, self.train_history['val_loss'], 
                    label='Validation Loss', linewidth=2, marker='s', markersize=3)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # print final  Loss
        print(f"\ntraining results:")
        print(f"  Final train  Loss:   {self.train_history['train_loss'][-1]:.6f}")
        if self.train_history['val_loss']:
            print(f"  Final valid Loss:   {self.train_history['val_loss'][-1]:.6f}")
            best_val_loss = min(self.train_history['val_loss'])
            best_epoch = self.train_history['val_loss'].index(best_val_loss) + 1
            print(f"  Best valid Loss:   {best_val_loss:.6f} (Epoch {best_epoch})")
    # Scoring function
    def score(self, X, update_ewma_state=True):
        """Return d, EWMA(d),  recon_error"""
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        self.ae.eval()
        with torch.no_grad():
            X_recon, Z = self.ae(X_t)
            Z = Z.cpu().numpy()
            re =  torch.mean((X_t - X_recon) ** 2, dim=1).cpu().numpy()
            re = np.log1p(re)
        # Mahalanobis Distance
        d2 = self.mcd.mahalanobis(Z)
        d = np.sqrt(np.maximum(d2, 0.0)) 
        # EWMA
        ew = self._ewma(d, self.ewma_lambda, init_value=self.ewma_init)
        if update_ewma_state:
            self.ewma_init = ew[-1]  
        return d, ew, re     

            
    # prediction function
    def predict(self, X, logic='or', return_details=False):
        """
        predict anomaly labels based on three criteria 
        """
        d, ew, re = self.score(X, update_ewma_state=False)
        
        # three logics flag
        flag_shewhart = (d > self.ucl_shewhart)
        flag_ewma = (ew > self.ucl_ewma)
        flag_re = (re > self.ucl_re) if self.use_re else np.zeros_like(d, dtype=bool)
        
        if logic == 'or':
            pred = (flag_shewhart | flag_ewma | flag_re).astype(int)
        elif logic == 'and':
            if self.use_re:
                pred = (flag_shewhart & flag_ewma & flag_re).astype(int)
            else:
                pred = (flag_shewhart & flag_ewma).astype(int)
        elif logic == 'vote':  # 2 out of 3
            votes = flag_shewhart.astype(int) + flag_ewma.astype(int)
            if self.use_re:
                votes += flag_re.astype(int)
                pred = (votes >= 2).astype(int)
            else:
                pred = (votes >= 2).astype(int)
        else:
            raise ValueError(f"Unknown logic: {logic}")
        
        if return_details:
            return pred, d, ew, re
        return pred 
    # Anomaly explanation function    
    def explain_anomaly(self, x_anomaly, topk=5, feature_names=None):
        self.ae.eval()
        x = np.asarray(x_anomaly, dtype=np.float32).reshape(1, -1)
        X_t = torch.from_numpy(x).to(self.device)

    # ---- latent z ----
        with torch.no_grad():
            z = self.ae.encode(X_t).detach().cpu().numpy()[0]

        mu = self.mcd.location_
        cov = self.mcd.covariance_
        diff = (z - mu)

    # ---- latent contribution: u_i^2 where u = L (z-mu), L = chol(inv(cov)) ----
        cov_inv = np.linalg.pinv(cov)
        L = np.linalg.cholesky(cov_inv + 1e-12 * np.eye(cov_inv.shape[0]))
        u = L @ diff
        latent_contrib = u**2  # sums to d^2

    # print latent topk
        idx = np.argsort(latent_contrib)[::-1][:topk]
        print("Latent contribution (to d^2), top:", topk)
        for i in idx:
            print(f"  z[{i}] = {latent_contrib[i]:.4f}")

    # ---- input feature importance (gradient-based) ----
    
        X_t2 = X_t.clone().detach().requires_grad_(True)
        z_t = self.ae.encoder(X_t2)        # (1, latent_dim)

        grads = []
        for j in range(z_t.shape[1]):
            if X_t2.grad is not None:
                X_t2.grad.zero_()
            z_t[0, j].backward(retain_graph=True)
            grads.append(X_t2.grad.detach().abs().cpu().numpy()[0])

    # importance: mean |dz/dx|
        input_importance = np.mean(np.stack(grads, axis=0), axis=0)
        #  print input feature topk
        names = feature_names if feature_names is not None else [f"feat_{k}" for k in range(len(input_importance))]
        top_idx = np.argsort(input_importance)[::-1][:topk]
        print("\nInput feature importance (|dz/dx| mean), top:", topk)
        for k in top_idx:
            print(f"  {names[k]}: {input_importance[k]:.4f}")

        return latent_contrib, input_importance
    

    def plot_latent_space(self, X_normal, X_anomaly=None):
        from sklearn.decomposition import PCA

        self.ae.eval()
        with torch.no_grad():
            Z_normal = self.ae.encode(torch.from_numpy(X_normal.astype(np.float32)).to(self.device)).detach().cpu().numpy()

        if Z_normal.shape[1] > 2:
            pca = PCA(n_components=2)
            Z_2d = pca.fit_transform(Z_normal)
            explained_var = pca.explained_variance_ratio_
        else:
            pca = None
            Z_2d = Z_normal
            explained_var = [1.0, 1.0]

        plt.figure(figsize=(10, 8))
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], alpha=0.4, label='Normal', s=20)

        if X_anomaly is not None:
            with torch.no_grad():
                Z_anom = self.ae.encode(torch.from_numpy(X_anomaly.astype(np.float32)).to(self.device)).detach().cpu().numpy()
            Z_anom_2d = pca.transform(Z_anom) if pca is not None else Z_anom
            plt.scatter(Z_anom_2d[:, 0], Z_anom_2d[:, 1], alpha=0.8, label='Anomaly', s=60, marker='x')
  
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)' if pca else 'Latent Dim 1')
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)' if pca else 'Latent Dim 2')
        plt.title('Latent Space Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_reconstruction(self, X_samples, n_examples=5, feature_names=None):
        """Plot original vs reconstructed samples"""
        self.ae.eval()
        X_samples = np.asarray(X_samples, dtype=np.float32)[:n_examples]
        with torch.no_grad():
            X_t = torch.from_numpy(X_samples).to(self.device)
            X_recon, _ = self.ae(X_t)
            X_recon = X_recon.cpu().numpy()
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))
        if n_examples == 1:
            axes = [axes]
        
        x_axis = np.arange(X_samples.shape[1])
        for i, ax in enumerate(axes):
            ax.plot(x_axis, X_samples[i], label='Original', alpha=0.7)
            ax.plot(x_axis, X_recon[i], label='Reconstructed', 
                   alpha=0.7, linestyle='--')
            ax.set_title(f'Sample {i+1}')
            if feature_names is not None:
                ax.set_xticks(x_axis)
                ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()      
