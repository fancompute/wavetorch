import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from wavetorch.core import load_model

mpl.rcParams['text.usetex'] = True

COL_TRAIN = "#1f77b4"
COL_TEST = "#2ca02c"

files = ['./study/nonlinear_speed/kerr_264_cv.pt',
         './study/nonlinear_speed/kerr_587_cv.pt',
         './study/nonlinear_speed/kerr_588_cv.pt',
         './study/nonlinear_speed/kerr_589_cv.pt',
         './study/nonlinear_speed/kerr_590_cv.pt',
         './study/nonlinear_speed/kerr_599_cv.pt',
         ]

num_dof = []
acc_train_mean = []
acc_train_std = []
acc_test_mean = []
acc_test_std = []

for file in files:
    model, history, history_state, cfg = load_model(file)

    history_mean = history.groupby('epoch').mean()
    history_std = history.groupby('epoch').std()

    acc_train_mean.append(history_mean['acc_train'].tail(1).item() * 100)
    acc_train_std.append(history_std['acc_train'].tail(1).item() * 100)
    acc_test_mean.append(history_mean['acc_test'].tail(1).item() * 100)
    acc_test_std.append(history_std['acc_test'].tail(1).item() * 100)

    num_dof.append(model.design_region.sum().item())

inds = np.argsort(num_dof)

num_dof = np.array(num_dof)[inds]
acc_train_mean = np.array(acc_train_mean)[inds]
acc_train_std = np.array(acc_train_std)[inds]
acc_test_mean = np.array(acc_test_mean)[inds]
acc_test_std = np.array(acc_test_std)[inds]

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 2.25))
ax.plot(num_dof, acc_train_mean, 'o-', color=COL_TRAIN, label="Training dataset")
ax.fill_between(num_dof, acc_train_mean - acc_train_std, acc_train_mean + acc_train_std, color=COL_TRAIN, alpha=0.15)
ax.plot(num_dof, acc_test_mean, 'o-', color=COL_TEST, label="Testing dataset")
ax.fill_between(num_dof, acc_test_mean - acc_test_std, acc_test_mean + acc_test_std, color=COL_TEST, alpha=0.15)
ax.set_xlabel("Trainable degrees of freedom")
ax.set_ylabel("Accuracy")

ax.legend()

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10))
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f\%%'))
ax.set_ylim([20, 100])

plt.show()
