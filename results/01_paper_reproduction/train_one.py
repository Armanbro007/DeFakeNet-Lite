
import os, sys, json, time, argparse, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.applications import ResNet50V2
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, f1_score, precision_score, recall_score)

ap = argparse.ArgumentParser()
for k in ["variant", "cache", "outdir"]: ap.add_argument("--" + k, required=True)
for k in ["seed", "epochs", "batch", "img", "es_patience", "lr_patience", "workers"]:
    ap.add_argument("--" + k, type=int, required=True)
for k in ["lr", "lr_factor", "min_lr"]: ap.add_argument("--" + k, type=float, required=True)
a = ap.parse_args()
key = f"DeFakeNet_paper__{a.variant}__s{a.seed}"
freeze_bn = {"bn_trainable": False, "bn_frozen": True}[a.variant]

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)

# ---- data: 224x224, [0,1] scaling, paper's augmentation (Sec. 3.1) ----
x_tr = np.load(f"{a.cache}/x_train.npy"); y_tr = np.load(f"{a.cache}/y_train.npy")
x_va = np.load(f"{a.cache}/x_val.npy").astype("float32") / 255.0; y_va = np.load(f"{a.cache}/y_val.npy")
x_te = np.load(f"{a.cache}/x_test.npy").astype("float32") / 255.0; y_te = np.load(f"{a.cache}/y_test.npy")
train_aug = ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=10, width_shift_range=0.2,
                               height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
train_it = train_aug.flow(x_tr, y_tr, batch_size=a.batch, shuffle=True, seed=a.seed)
del x_tr

# ---- model: Table 3 / Sec. 3.2 ----
inp = layers.Input(shape=(a.img, a.img, 3), name="InputLayer")
base = ResNet50V2(include_top=False, weights="imagenet", input_shape=(a.img, a.img, 3))
base.trainable = True                                   # the entire base model is fine-tuned
if freeze_bn:                                           # Sec. 3.2 sentence: BatchNorm kept frozen
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
x = base(inp)
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
x = layers.Dense(1024, activation="relu", name="dense")(x)
out = layers.Dense(1, activation="sigmoid", name="dense_1")(x)
model = keras.Model(inp, out, name="DeFakeNet")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=a.lr),
              loss="binary_crossentropy", metrics=["accuracy"])
n_tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
n_all = model.count_params()
print(f"[{key}] params total {n_all:,} | trainable {n_tr:,} | non-trainable {n_all - n_tr:,} | "
      f"layers {len(base.layers) + len(model.layers)}", flush=True)

# ---- training: Sec. 3.3 ----
ckpt = f"{a.outdir}/{key}_best.weights.h5"
cbs = [EarlyStopping(monitor="val_loss", patience=a.es_patience, restore_best_weights=True, verbose=1),
       ReduceLROnPlateau(monitor="val_loss", factor=a.lr_factor, patience=a.lr_patience,
                         min_lr=a.min_lr, verbose=1),
       ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
       CSVLogger(f"{a.outdir}/{key}_epochs.csv")]
t0 = time.time()
hist = model.fit(train_it, validation_data=(x_va, y_va), validation_batch_size=a.batch, epochs=a.epochs,
                 callbacks=cbs, workers=a.workers, use_multiprocessing=False, verbose=2)
train_s = time.time() - t0
h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
model.load_weights(ckpt)                                # weights of the lowest validation loss

# ---- test: threshold 0.5, Real = positive (Sec. 4) ----
p = model.predict(x_te, batch_size=a.batch, verbose=0).ravel()
yt = y_te.astype(int); yp = (p >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
t1 = time.time()
for _ in range(5): model.predict(x_te[:320], batch_size=a.batch, verbose=0)
ms = (time.time() - t1) / (5 * 320) * 1000

best = int(np.argmin(h["val_loss"]))
lrs = h.get("lr", h.get("learning_rate", [a.lr] * len(h["loss"])))
cut = next((i + 1 for i, v in enumerate(lrs) if v < a.lr * 0.99), None)
res = {
    "key": key, "variant": a.variant, "seed": a.seed, "freeze_bn": freeze_bn,
    "params_total": int(n_all), "params_trainable": n_tr, "params_non_trainable": int(n_all - n_tr),
    "epochs_run": len(h["loss"]), "train_seconds": round(train_s, 1),
    "run": {"train_acc_ep1": round(h["accuracy"][0] * 100, 2),
            "train_acc_ep12": round(h["accuracy"][11] * 100, 2) if len(h["accuracy"]) >= 12 else None,
            "val_acc_ep12": round(h["val_accuracy"][11] * 100, 2) if len(h["val_accuracy"]) >= 12 else None,
            "first_lr_cut_epoch": cut, "best_epoch": best + 1,
            "best_val_loss": round(h["val_loss"][best], 4),
            "val_acc_at_best": round(h["val_accuracy"][best] * 100, 2),
            "stopped_epoch": len(h["loss"]), "final_train_acc": round(h["accuracy"][-1] * 100, 2)},
    "test": {"acc": round(100 * (tp + tn) / len(yt), 2), "auc": round(100 * roc_auc_score(yt, p), 2),
             "pr_auc": round(100 * average_precision_score(yt, p), 2),
             "f1": round(100 * f1_score(yt, yp), 2), "mcc": round(100 * matthews_corrcoef(yt, yp), 2),
             "precision": round(100 * precision_score(yt, yp), 2), "recall": round(100 * recall_score(yt, yp), 2),
             "specificity": round(100 * tn / (tn + fp), 2),
             "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "ms_per_img": round(ms, 2), "history": h,
}
np.savez_compressed(f"{a.outdir}/{key}_probs.npz", probs=p, y=yt)
json.dump(res, open(f"{a.outdir}/{key}_results.json", "w"), indent=1)
print(f"[{key}] DONE test acc {res['test']['acc']:.2f} | AUC {res['test']['auc']:.2f} | "
      f"best epoch {best + 1} | stopped {len(h['loss'])}", flush=True)
