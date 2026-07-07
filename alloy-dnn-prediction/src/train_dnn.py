# ============================================================
#  Deep Neural Network — Alloy Mechanical Property Prediction
#  Google Colab Ready Script
#  Project 23 | IMI Project | pmo.csv Dataset
# ============================================================

# ── STEP 0: Install / Import ─────────────────────────────────
# Run this cell first in Colab
# !pip install shap --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("✅ All libraries imported successfully!")
print(f"   TensorFlow version : {tf.__version__}")


# ════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD & EXPLORE DATA
# ════════════════════════════════════════════════════════════

# ── Upload pmo.csv in Colab ──────────────────────────────────
# from google.colab import files
# uploaded = files.upload()          # click "Choose Files" → select pmo.csv
# df = pd.read_csv('pmo.csv')

# ── OR if you mount Google Drive ────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/MyDrive/pmo.csv')

# ── For local / already-uploaded file ───────────────────────
df = pd.read_csv('pmo.csv')          # change path if needed

# Strip any extra whitespace in column names
df.columns = df.columns.str.strip()

print("\n📊 Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nFirst 5 Rows:")
df.head()


# ════════════════════════════════════════════════════════════
#  SECTION 2 — DATA PREPROCESSING
# ════════════════════════════════════════════════════════════

# 2-A: Drop non-numeric identifier column
df_model = df.drop(columns=['Alloy code'])

# 2-B: Check for missing values
print("\n🔍 Missing Values per Column:")
print(df_model.isnull().sum())

# 2-C: Basic statistics
print("\n📈 Descriptive Statistics:")
print(df_model.describe().T.round(4))

# 2-D: Define Features (X) and Targets (y)
FEATURE_COLS = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu',
                'V', 'Al', 'N', 'Ceq', 'Nb + Ta', 'Temperature (°C)']
TARGET_COLS  = ['0.2% Proof Stress (MPa)', 'Tensile Strength (MPa)']

# Handle encoding issue in Temperature column name
temp_col = [c for c in df_model.columns if 'Temperature' in c or 'Temp' in c][0]
FEATURE_COLS[-1] = temp_col   # use actual column name

X = df_model[FEATURE_COLS].values
y = df_model[TARGET_COLS].values

print(f"\n✅ Features shape : {X.shape}")
print(f"✅ Targets shape  : {y.shape}")

# 2-E: Train / Test Split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2-F: Feature Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train)
y_test_sc  = scaler_y.transform(y_test)

print("\n✅ Data split and scaled.")
print(f"   Train : {X_train_sc.shape}  |  Test : {X_test_sc.shape}")


# ════════════════════════════════════════════════════════════
#  SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Target Variable Distributions', fontsize=15, fontweight='bold')

for ax, col in zip(axes, TARGET_COLS):
    ax.hist(df_model[col], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel('MPa')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = df_model.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Temperature vs Targets
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Temperature vs Mechanical Properties', fontsize=13, fontweight='bold')
colors = df_model[temp_col]

for ax, col in zip(axes, TARGET_COLS):
    sc = ax.scatter(df_model[temp_col], df_model[col],
                    c=colors, cmap='plasma', alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label='Temperature (°C)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(col)
    ax.set_title(col)

plt.tight_layout()
plt.savefig('temp_vs_properties.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ EDA plots saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 4 — BUILD THE DEEP NEURAL NETWORK
# ════════════════════════════════════════════════════════════

def build_dnn(input_dim, output_dim=2, learning_rate=0.001):
    """
    Deep Neural Network for multi-output regression.
    Architecture: Input → 128 → 256 → 128 → 64 → Output
    """
    inputs = keras.Input(shape=(input_dim,), name='composition_temperature')

    x = layers.Dense(128, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(64, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    outputs = layers.Dense(output_dim, name='mechanical_properties')(x)

    model = Model(inputs, outputs, name='DNN_Alloy')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_dnn(input_dim=X_train_sc.shape[1])
model.summary()


# ════════════════════════════════════════════════════════════
#  SECTION 5 — TRAIN THE MODEL
# ════════════════════════════════════════════════════════════

cb_early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True, verbose=1
)
cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1
)
cb_checkpoint = callbacks.ModelCheckpoint(
    'best_dnn_alloy.h5', monitor='val_loss', save_best_only=True, verbose=0
)

history = model.fit(
    X_train_sc, y_train_sc,
    validation_split=0.15,
    epochs=500,
    batch_size=32,
    callbacks=[cb_early_stop, cb_reduce_lr, cb_checkpoint],
    verbose=1
)

print(f"\n✅ Training complete. Best epoch: {np.argmin(history.history['val_loss'])+1}")


# ════════════════════════════════════════════════════════════
#  SECTION 6 — TRAINING CURVES
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DNN Training History', fontsize=14, fontweight='bold')

axes[0].plot(history.history['loss'], label='Train Loss', color='royalblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='tomato')
axes[0].set_title('MSE Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_yscale('log')

axes[1].plot(history.history['mae'], label='Train MAE', color='royalblue')
axes[1].plot(history.history['val_mae'], label='Val MAE', color='tomato')
axes[1].set_title('Mean Absolute Error')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE (scaled)')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════
#  SECTION 7 — EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════

def evaluate_model(name, y_true, y_pred, target_names=TARGET_COLS):
    """Compute and print R², MAE, RMSE, MAPE for each target."""
    results = {}
    print(f"\n{'='*55}")
    print(f"  📊 {name}")
    print(f"{'='*55}")
    for i, tname in enumerate(target_names):
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        results[tname] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        print(f"\n  🎯 {tname}")
        print(f"     R²   = {r2:.4f}")
        print(f"     MAE  = {mae:.2f} MPa")
        print(f"     RMSE = {rmse:.2f} MPa")
        print(f"     MAPE = {mape:.2f}%")
    return results


# ════════════════════════════════════════════════════════════
#  SECTION 8 — DNN PREDICTIONS & PLOTS
# ════════════════════════════════════════════════════════════

# Predict and inverse-transform back to original MPa scale
y_pred_sc  = model.predict(X_test_sc)
y_pred_dnn = scaler_y.inverse_transform(y_pred_sc)

dnn_results = evaluate_model("Deep Neural Network (DNN)", y_test, y_pred_dnn)

# Actual vs Predicted Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('DNN — Actual vs Predicted', fontsize=14, fontweight='bold')

for i, (ax, col) in enumerate(zip(axes, TARGET_COLS)):
    ax.scatter(y_test[:, i], y_pred_dnn[:, i],
               alpha=0.6, color='steelblue', edgecolors='white', s=40)
    mn = min(y_test[:, i].min(), y_pred_dnn[:, i].min())
    mx = max(y_test[:, i].max(), y_pred_dnn[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
    r2 = dnn_results[col]['R2']
    ax.set_title(f'{col}\nR² = {r2:.4f}')
    ax.set_xlabel('Actual (MPa)')
    ax.set_ylabel('Predicted (MPa)')
    ax.legend()

plt.tight_layout()
plt.savefig('dnn_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# Residual Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DNN — Residual Analysis', fontsize=14, fontweight='bold')

for i, (ax, col) in enumerate(zip(axes, TARGET_COLS)):
    residuals = y_test[:, i] - y_pred_dnn[:, i]
    ax.scatter(y_pred_dnn[:, i], residuals, alpha=0.5, color='darkorange', s=30)
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Predicted (MPa)')
    ax.set_ylabel('Residual (MPa)')
    ax.set_title(col)

plt.tight_layout()
plt.savefig('dnn_residuals.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════
#  SECTION 9 — BASELINE MODELS (For Comparison)
# ════════════════════════════════════════════════════════════

# --- Linear Regression ---
lr = MultiOutputRegressor(LinearRegression())
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
lr_results = evaluate_model("Linear Regression", y_test, y_pred_lr)

# --- Random Forest ---
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_results = evaluate_model("Random Forest Regressor", y_test, y_pred_rf)


# ════════════════════════════════════════════════════════════
#  SECTION 10 — MODEL COMPARISON TABLE & CHART
# ════════════════════════════════════════════════════════════

models_list  = ['Linear Regression', 'Random Forest', 'DNN (Ours)']
results_list = [lr_results, rf_results, dnn_results]

for target in TARGET_COLS:
    print(f"\n{'─'*60}")
    print(f"  Comparison Table — {target}")
    print(f"{'─'*60}")
    print(f"  {'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print(f"  {'─'*56}")
    for mname, res in zip(models_list, results_list):
        r  = res[target]
        print(f"  {mname:<22} {r['R2']:>8.4f} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['MAPE']:>7.2f}%")

# Bar Chart Comparison
metrics = ['R2', 'MAE', 'RMSE']
colors  = ['#2196F3', '#4CAF50', '#FF5722']

fig, axes = plt.subplots(len(metrics), len(TARGET_COLS),
                         figsize=(14, 10), constrained_layout=True)
fig.suptitle('Model Comparison — DNN vs Baselines', fontsize=14, fontweight='bold')

for col_idx, target in enumerate(TARGET_COLS):
    for row_idx, metric in enumerate(metrics):
        ax = axes[row_idx][col_idx]
        vals = [res[target][metric] for res in results_list]
        bars = ax.bar(models_list, vals, color=colors, edgecolor='white', width=0.5)
        ax.set_title(f'{target}\n{metric}', fontsize=9)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002*max(vals),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', labelrotation=15, labelsize=8)

plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Comparison chart saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 11 — FEATURE IMPORTANCE (Random Forest Proxy)
# ════════════════════════════════════════════════════════════

# Use RF feature importances as a proxy for input sensitivity
feature_names = FEATURE_COLS
importance_ps  = rf.estimators_[0].feature_importances_
importance_ts  = rf.estimators_[1].feature_importances_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Importance (Random Forest Proxy)', fontsize=13, fontweight='bold')

for ax, imp, col in zip(axes, [importance_ps, importance_ts], TARGET_COLS):
    sorted_idx = np.argsort(imp)
    ax.barh([feature_names[i] for i in sorted_idx], imp[sorted_idx],
            color='steelblue', edgecolor='white')
    ax.set_title(col)
    ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════
#  SECTION 12 — K-FOLD CROSS VALIDATION (DNN)
# ════════════════════════════════════════════════════════════

print("\n🔁 Running 5-Fold Cross Validation on DNN...\n")

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
fold_r2 = {'Proof Stress': [], 'Tensile Strength': []}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    sc_X = StandardScaler(); sc_y = StandardScaler()
    X_tr_s  = sc_X.fit_transform(X_tr)
    X_val_s = sc_X.transform(X_val)
    y_tr_s  = sc_y.fit_transform(y_tr)

    fold_model = build_dnn(input_dim=X_tr_s.shape[1])
    fold_model.fit(
        X_tr_s, y_tr_s,
        validation_split=0.1,
        epochs=300, batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=25, restore_best_weights=True)
        ],
        verbose=0
    )

    y_val_pred = sc_y.inverse_transform(fold_model.predict(X_val_s))
    r2_ps = r2_score(y_val[:, 0], y_val_pred[:, 0])
    r2_ts = r2_score(y_val[:, 1], y_val_pred[:, 1])

    fold_r2['Proof Stress'].append(r2_ps)
    fold_r2['Tensile Strength'].append(r2_ts)
    print(f"  Fold {fold+1}: Proof Stress R² = {r2_ps:.4f} | Tensile Strength R² = {r2_ts:.4f}")

print(f"\n  📊 Cross-Validation R² Summary:")
for k, v in fold_r2.items():
    print(f"     {k}: Mean = {np.mean(v):.4f}  |  Std = {np.std(v):.4f}")

# Plot CV results
fig, ax = plt.subplots(figsize=(8, 5))
folds_x  = range(1, 6)
ax.plot(folds_x, fold_r2['Proof Stress'],    'o-', label='0.2% Proof Stress', color='royalblue', lw=2)
ax.plot(folds_x, fold_r2['Tensile Strength'],'s-', label='Tensile Strength',  color='tomato',    lw=2)
ax.axhline(np.mean(fold_r2['Proof Stress']),    color='royalblue', linestyle='--', alpha=0.5)
ax.axhline(np.mean(fold_r2['Tensile Strength']),color='tomato',    linestyle='--', alpha=0.5)
ax.set_title('5-Fold Cross Validation — R² per Fold', fontsize=13, fontweight='bold')
ax.set_xlabel('Fold')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kfold_cv_results.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════
#  SECTION 13 — SAVE MODEL & FINAL SUMMARY
# ════════════════════════════════════════════════════════════

model.save('dnn_alloy_final.h5')
print("\n✅ Final model saved as 'dnn_alloy_final.h5'")

print("\n" + "═"*55)
print("  ✅  FINAL SUMMARY — DNN MODEL PERFORMANCE")
print("═"*55)
for target in TARGET_COLS:
    r = dnn_results[target]
    print(f"\n  🎯 {target}")
    print(f"     R²   = {r['R2']:.4f}")
    print(f"     MAE  = {r['MAE']:.2f} MPa")
    print(f"     RMSE = {r['RMSE']:.2f} MPa")
    print(f"     MAPE = {r['MAPE']:.2f}%")
print("\n  📁 Saved Outputs:")
print("     • best_dnn_alloy.h5         — best weights")
print("     • dnn_alloy_final.h5        — final model")
print("     • target_distributions.png")
print("     • correlation_heatmap.png")
print("     • temp_vs_properties.png")
print("     • training_history.png")
print("     • dnn_actual_vs_predicted.png")
print("     • dnn_residuals.png")
print("     • model_comparison.png")
print("     • feature_importance.png")
print("     • kfold_cv_results.png")
print("═"*55)

# ── END OF SCRIPT ────────────────────────────────────────────
