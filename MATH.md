# Math

Notation. Applicant $i = 1, \dots, n$ has a label $y_i \in \{0, 1\}$ (1 = default) and a predicted probability $\hat p_i$. $P = \sum_i y_i$ is the number of defaulters, $N = n - P$ the number of non-defaulters, and $\pi = P / n$ the base rate. In `application_train`, $\pi = 24{,}825 / 307{,}511 = 0.0807$.

## Evaluation metrics

All metrics are computed by `src/evaluate.py`, written with NumPy and checked against scikit-learn and against the worked example below (`tests/test_evaluate.py`).

### Counts at a threshold

Flag every applicant with $\hat p_i \ge t$:

$$
TP(t) = \sum_i \mathbf{1}[\hat p_i \ge t,\ y_i = 1], \qquad
FP(t) = \sum_i \mathbf{1}[\hat p_i \ge t,\ y_i = 0]
$$

$$
\text{precision}(t) = \frac{TP(t)}{TP(t) + FP(t)}, \qquad
\text{recall}(t) = \frac{TP(t)}{P}
$$

Precision is set to 0 when nobody is flagged.

### ROC-AUC

$$
\text{TPR}(t) = \frac{TP(t)}{P}, \qquad \text{FPR}(t) = \frac{FP(t)}{N}
$$

ROC-AUC is the area under TPR plotted against FPR, by the trapezoid rule over the distinct scores. It equals the probability that a random defaulter gets a higher score than a random non-defaulter, counting a tie as one half:

$$
\text{ROC-AUC} = \Pr(\hat p_{+} > \hat p_{-}) + \tfrac{1}{2}\Pr(\hat p_{+} = \hat p_{-})
$$

It does not depend on the base rate. A random ranking gives 0.5.

### PR-AUC (average precision)

Sort the distinct scores $t_1 > t_2 > \dots > t_K$ and write $R_k = \text{recall}(t_k)$, $P_k = \text{precision}(t_k)$, $R_0 = 0$:

$$
\text{PR-AUC} = \sum_{k=1}^{K} (R_k - R_{k-1})\, P_k
$$

This is a step sum, not a trapezoid. A random ranking gives $\pi = 0.0807$, so PR-AUC is read against that floor and not against 0.5. It is the main metric here because it ignores the 92% of applicants who do not default and are easy to rank.

### Log loss

$$
\text{log loss} = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \ln \hat p_i + (1 - y_i) \ln (1 - \hat p_i) \Big]
$$

with $\hat p_i$ clipped to $[10^{-15},\ 1 - 10^{-15}]$. A certain wrong answer then costs $-\ln 10^{-15} = 34.5$ instead of infinity.

### Predicting the base rate for everyone

With $\hat p_i = \pi$ for all $i$ there is one distinct score. At that threshold every applicant is flagged, so $TP = P$, $FP = N$, precision $= \pi$ and recall $= 1$:

$$
\text{PR-AUC} = (1 - 0)\,\pi = \pi, \qquad \text{ROC-AUC} = \tfrac{1}{2}, \qquad
\text{log loss} = -\pi \ln \pi - (1 - \pi)\ln(1 - \pi)
$$

At $\pi = 0.0807$ the log loss is 0.2805.

### Worked example

Six applicants, highest score first:

| Applicant | $\hat p$ | $y$ |
|---|---|---|
| A | 0.9 | 1 |
| B | 0.8 | 1 |
| C | 0.7 | 0 |
| D | 0.6 | 1 |
| E | 0.4 | 0 |
| F | 0.2 | 0 |

- ROC-AUC: the defaulter scores higher in 3 + 3 + 2 = 8 of the 9 defaulter and non-defaulter pairs, so $8/9 = 0.8889$.
- PR-AUC: precision is 1, 1, 2/3, 3/4 at the thresholds where recall is 1/3, 2/3, 2/3, 1, so $\tfrac13 \cdot 1 + \tfrac13 \cdot 1 + 0 + \tfrac13 \cdot \tfrac34 = 11/12 = 0.9167$.
- Log loss: $-\tfrac{1}{6}(\ln 0.9 + \ln 0.8 + \ln 0.3 + \ln 0.6 + \ln 0.6 + \ln 0.8) = 0.4629$.
- At $t = 0.5$ the flagged are A, B, C, D, so precision is $3/4$ and recall is $3/3$.

## Baselines

Two models set the floor. The base-rate model predicts $\hat p_i = \pi$ for everyone. The second model is a logistic regression on one feature, the mean of the available values among `EXT_SOURCE_1`, `EXT_SOURCE_2` and `EXT_SOURCE_3`:

$$
\hat p_i = \sigma(\beta_0 + \beta_1 x_i), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The 172 applicants with none of the three scores get the median, fitted on the training fold only.

Scores are 5-fold stratified cross-validation on the 246,008 training applicants (seed 42), mean over folds with the standard deviation in brackets. The holdout is not used.

| Model | PR-AUC | ROC-AUC | Log loss |
|---|---|---|---|
| Base rate for everyone | 0.0807 | 0.5000 | 0.2805 |
| `EXT_SOURCE` mean, logistic | 0.1889 (0.0021) | 0.7164 (0.0025) | 0.2577 (0.0004) |

![ROC and precision-recall curves of the two baselines](docs/img/baseline_curves.png)

The curves pool the out-of-fold predictions of the 5 folds. The base-rate model is the diagonal on the ROC curve and the horizontal line at $\pi$ on the precision-recall curve.

The one-feature model lifts PR-AUC to 2.3 times the base rate and lowers log loss from 0.2805 to 0.2577. Its precision falls from about 0.3 near the top of the ranking to 0.08 at full recall. Every later model is compared with these two rows.

## Linear regression and squared loss

Side problem: predict the loan amount `AMT_CREDIT` from the applicant's other application columns. `AMT_GOODS_PRICE` (correlation 0.987 with `AMT_CREDIT`) and `AMT_ANNUITY` are left out because they are near copies of the target. Columns are preprocessed as for the default model, with 117 raw columns becoming about 300 after encoding. Scores are 5-fold cross-validation on the 246,008 training applicants (`src/linear_regression.py`).

### Model and loss

With a design matrix $X$ ($n$ rows, a leading column of ones for the intercept), targets $y$ and weights $w$:

$$
\hat y = X w, \qquad
L(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat y_i)^2 = \frac{1}{n} \lVert y - X w \rVert_2^2
$$

The gradient is

$$
\nabla_w L = -\frac{2}{n} X^\top (y - X w)
$$

Setting it to zero gives the normal equation

$$
X^\top X\, \hat w = X^\top y \quad\Longrightarrow\quad \hat w = (X^\top X)^{-1} X^\top y
$$

At the solution $X^\top (y - X\hat w) = 0$: the residuals sum to zero and are uncorrelated with every column. `tests/test_linear_regression.py` checks the closed form against scikit-learn and checks this orthogonality.

Metrics: $\text{RMSE} = \sqrt{L(\hat w)}$, $\text{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat y_i \rvert$ and

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}
$$

so $R^2 = 0$ is the score of predicting the mean $\bar y$ for everyone and negative values are worse than that.

### Ridge and lasso

Ridge adds the squared size of the weights (the intercept is not penalized):

$$
\hat w_{\text{ridge}} = \arg\min_w \; \lVert y - X w \rVert_2^2 + \alpha \lVert w \rVert_2^2
\quad\Longrightarrow\quad
\hat w_{\text{ridge}} = (X^\top X + \alpha I)^{-1} X^\top y \quad \text{(centered columns)}
$$

Lasso adds the absolute size, which has no closed form and sets weights exactly to zero:

$$
\hat w_{\text{lasso}} = \arg\min_w \; \frac{1}{2n} \lVert y - X w \rVert_2^2 + \alpha \lVert w \rVert_1
$$

scikit-learn's two $\alpha$ are on different scales, since ridge uses the sum of squares and lasso the mean. The same grid cannot be used for both. The ridge closed form is in `src/linear_regression.py` and matches scikit-learn's `Ridge` to 6 digits in the tests, for $\alpha$ from 0 to $10^4$.

### Results

| Model | RMSE | MAE | $R^2$ |
|---|---|---|---|
| Predict the mean for everyone | 402,725 | | 0 |
| OLS, no clipping | 419,414 (fold std 124,402) | 279,489 | -0.16 |
| OLS, clipped | 347,500 (3,006) | 266,269 | 0.255 |
| Ridge, clipped, $\alpha = 100$ | 345,891 (956) | 266,152 | 0.262 |
| Lasso, clipped, $\alpha = 100$ | 345,856 (998) | 266,094 | 0.263 |

![Ridge and lasso against penalty strength, and out-of-fold residuals](docs/img/regression.png)

- **One applicant breaks the unclipped fit.** A single validation applicant has an income of 117,000,000 (1,066 standard deviations above the mean). The model multiplies it by its income weight and predicts a loan of 119.9 million for an actual 562,491. That one squared error is about 70% of the fold's total, which gives that fold an RMSE of 641,945 against about 363,000 in the other four. MAE hardly moves (272,373 against 280,000 to 282,000) because it grows linearly with the error and not with its square.
- **Clipping fixes it.** Every numeric column with more than two distinct values is capped at the 0.1th and 99.9th percentile of the training fold. Binary flags are left alone, since a flag set for 2 applicants would otherwise be clipped away. RMSE falls to 347,500 on every fold and $R^2$ rises to 0.255, because the extreme values also distorted the weights for all other applicants.
- **With clipping the penalty helps slightly.** Ridge and lasso at $\alpha = 100$ improve RMSE by about 1,600 over OLS, more than the fold spread of about 1,000. Lasso at $\alpha = 100$ keeps 181 of 296 columns for the same score. Large penalties hurt: ridge at $\alpha = 10^6$ gives RMSE 384,053.
- **Residuals.** In 12 bins of predicted amount the mean residual stays within about 46,000 of 0, small against the spread, so the fit is not biased. The spread grows with the predicted amount: the 10th to 90th percentile of the residual is -140,000 to +181,000 around a prediction of 234,000 and -644,000 to +731,000 around 1,031,000. The diagonal stripes are loan amounts that repeat (round figures). 14 of the 10,000 plotted predictions are negative.
- The profile of an applicant explains only 26% of the variance of the loan amount.

## Logistic regression

The default model: a linear score turned into a probability. The preprocessed training table has $n = 246{,}008$ rows and $d = 301$ columns (`src/preprocess.py`, standardized). Cross-validation results are in `results/logistic.csv` and `results/regularization_curve.csv`.

### Model

For applicant $i$ with column vector $x_i$, weights $w$ and intercept $b$:

$$
z_i = b + x_i^\top w, \qquad \hat p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}, \qquad
\ln \frac{\hat p_i}{1 - \hat p_i} = z_i
$$

The score $z_i$ is the log-odds of default. $\sigma$ maps it into $(0, 1)$, with $\sigma(0) = 0.5$, $\sigma(-z) = 1 - \sigma(z)$ and $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$. Increasing a standardized column by one standard deviation multiplies the odds of default by $e^{w_j}$: with the fitted weights of the chosen model, $e^{-0.489} = 0.61$ for `EXT_SOURCE_3` and $e^{-0.376} = 0.69$ for `EXT_SOURCE_2`.

![The sigmoid and the out-of-fold scores of 20,000 training applicants](docs/img/logistic_sigmoid.png)

The real scores sit on the flat lower part of the curve. The median score is -2.90 for applicants who did not default and -1.99 for those who did, a difference of 0.91 log-odds (odds 2.5 times higher). The two distributions overlap heavily, which is what a ROC-AUC of 0.75 looks like. Only 0.17% of applicants have a score above 0 (probability above 0.5), so a threshold of 0.5 would flag almost nobody.

### Loss and gradient

The weights maximize the Bernoulli likelihood of the labels, which is the same as minimizing the log loss of the evaluation section:

$$
L(b, w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i, \qquad
\ell_i = -y_i \ln \hat p_i - (1 - y_i) \ln (1 - \hat p_i)
$$

Since $\sigma' = \sigma(1 - \sigma)$, the derivative with respect to the score is

$$
\frac{\partial \ell_i}{\partial z_i} = -y_i (1 - \hat p_i) + (1 - y_i)\,\hat p_i = \hat p_i - y_i
$$

and the chain rule through $z_i = b + x_i^\top w$ gives

$$
\nabla_w L = \frac{1}{n} X^\top (\hat p - y), \qquad
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat p_i - y_i)
$$

This has the same form as the squared-loss gradient $-\tfrac{2}{n} X^\top (y - \hat y)$: a column's weight moves in proportion to how that column correlates with the miss. The Hessian is $\tfrac{1}{n} X^\top \mathrm{diag}\big(\hat p_i (1 - \hat p_i)\big) X$, which is positive semi-definite, so $L$ is convex and has no local minima apart from the global one. At $w = 0$, $b = 0$ every probability is 0.5 and $L = \ln 2 = 0.6931$. `tests/test_logistic_math.py` checks the gradient against finite differences and on a two-applicant example worked out by hand.

### Regularization

scikit-learn minimizes, with the intercept left out of the penalty,

$$
\min_{b, w}\; C \sum_{i=1}^{n} \ell_i + \tfrac{1}{2} \lVert w \rVert_2^2 \quad (\text{L2}),
\qquad
\min_{b, w}\; C \sum_{i=1}^{n} \ell_i + \lVert w \rVert_1 \quad (\text{L1})
$$

Dividing by $C n$ puts it on the scale of the mean loss, with penalty weight $\lambda = 1 / (C n)$:

$$
\frac{1}{n} \sum_i \ell_i + \frac{\lambda}{2} \lVert w \rVert_2^2, \qquad
\nabla_w = \frac{1}{n} X^\top (\hat p - y) + \lambda w
$$

The L1 term adds $\lambda\,\mathrm{sign}(w_j)$ to the gradient of weight $j$ and sets weights that do not earn their place exactly to zero. A small $C$ is a large $\lambda$. For the chosen $C = 0.03$ and $n = 246{,}008$, $\lambda = 1.355 \times 10^{-4}$.

Cross-validated PR-AUC against $C$ is flat above $C = 0.01$ for L2 and the training and validation curves are 0.0044 apart at $C = 0.03$, so the model does not overfit. Weights of near-duplicate columns (`AMT_CREDIT` and `AMT_GOODS_PRICE`, correlation 0.987) take large opposite values that cancel.

![Validation curve for the L1 and L2 penalties](docs/img/validation_curve.png)

![Regularization path of the six largest weights](docs/img/regularization_path.png)

In the path chart, the L1 panel stops at $C = 0.1$ because a single L1 fit at $C = 1$ took 381 s with `liblinear`.

### Learning: gradient descent

Gradient descent repeats $w \leftarrow w - \eta\,\nabla_w$ and $b \leftarrow b - \eta\,\partial L / \partial b$. The step size $\eta = 1 / (\tfrac{1}{4}\lambda_{\max}(X^\top X / n) + \lambda)$ is the reciprocal of a bound on the curvature, and is 0.0986 here ($\lambda_{\max} = 39.6$). `src/logistic_math.py` implements the loss, the gradient and this loop in a few lines, and `src/logistic_demo.py` runs it on the real training table.

![Training objective of gradient descent against the scikit-learn solution](docs/img/logistic_loss_curve.png)

- The objective falls from 0.6931 to 0.2716 after 25 steps, 0.2500 after 1,000 and 0.2493 after 5,000, against 0.2488 for scikit-learn's solution (gap 0.00045).
- The predicted probabilities of the two solutions have a correlation of 0.993, but the weights only 0.88. Near-duplicate columns form a flat valley in which many weight vectors have almost the same loss, and plain gradient descent moves along it slowly. scikit-learn's L-BFGS uses curvature and needed about 100 to 150 iterations at $C$ from 0.01 to 1.
- On small problems (`tests/test_logistic_math.py`, 20,000 steps) the hand-written loop reaches scikit-learn's weights to within 0.002.

### Results

5-fold stratified cross-validation on the 246,008 training applicants, mean (standard deviation over folds):

| Model | PR-AUC | ROC-AUC | Log loss |
|---|---|---|---|
| Base rate for everyone | 0.0807 | 0.5000 | 0.2805 |
| `EXT_SOURCE` mean, logistic | 0.1889 (0.0021) | 0.7164 (0.0025) | 0.2577 (0.0004) |
| Logistic regression, all columns, L2, $C = 0.03$ | 0.2249 (0.0045) | 0.7475 (0.0012) | 0.2495 (0.0003) |

The holdout is not used yet.

## Decision tree

A decision tree splits the applicants with a yes/no question on one column, repeats inside each part, and predicts for a leaf the share of defaulters among the training applicants in it. Preprocessing is the same pipeline as for the linear models, without scaling. The settings are scikit-learn's defaults except `max_depth`, with the Gini criterion and no minimum leaf size.

### Depth and overfitting

`max_depth` from 1 to 30 and with no limit, 5-fold stratified cross-validation on the 246,008 training applicants. Mean over folds, validation fold:

| `max_depth` | Leaves | PR-AUC | ROC-AUC | Log loss |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.1076 | 0.5869 | 0.2732 |
| 3 | 8 | 0.1566 | 0.6826 | 0.2629 |
| 5 | 32 | 0.1794 | 0.7059 | 0.2590 |
| 7 | 122 | 0.1866 | 0.7138 | 0.2659 |
| 10 | 572 | 0.1779 | 0.7044 | 0.3680 |
| 15 | 2,537 | 0.1507 | 0.6556 | 0.9148 |
| 30 | 11,870 | 0.1014 | 0.5173 | 3.7271 |
| none (depth 56 to 63) | 15,993 | 0.0904 | 0.5374 | 5.1014 |

![Training and validation PR-AUC and log loss against tree depth](docs/img/tree_depth.png)

- Validation PR-AUC peaks at depth 7 (0.1866, std 0.0053 over folds) and falls from there, while the training PR-AUC rises to 1.0 for the unlimited tree.
- Log loss is best at depth 5 (0.2590) and is worse than the 0.2805 of predicting the base rate from depth 8 on (0.2834), because a deep tree gives leaves with a few applicants probabilities close to 0 or 1.
- The best tree is below the logistic regression (PR-AUC 0.2249) and just below the `EXT_SOURCE` mean baseline (0.1889).

## Random forest

A random forest fits many decision trees, each on a bootstrap sample of the training rows and with only a random subset of the columns tried at each split, and predicts the mean of the trees' leaf probabilities. All forests here use the Gini criterion and the same pipeline as above on 300 columns. Settings were swept one at a time on the same 5 folds, validation fold, mean (std over folds):

The three sweeps ran in this order and each used the best setting of the one before: leaf size first (100 trees, 17 columns per split), then columns per split (100 trees, `min_samples_leaf` = 30), then the number of trees (`min_samples_leaf` = 30, 30 columns).

| `min_samples_leaf` | Train PR-AUC | Validation PR-AUC | Validation log loss |
|---:|---:|---:|---:|
| 1 | 1.0000 | 0.1967 (0.0022) | 0.2711 (0.0024) |
| 10 | 0.7741 | 0.2193 (0.0033) | 0.2538 (0.0001) |
| 30 | 0.4469 | 0.2211 (0.0024) | 0.2542 (0.0004) |
| 100 | 0.2903 | 0.2172 (0.0021) | 0.2558 (0.0003) |
| 1000 | 0.2163 | 0.2049 (0.0019) | 0.2614 (0.0003) |

| `max_features` | Validation PR-AUC | Validation ROC-AUC | Validation log loss |
|---:|---:|---:|---:|
| 1 | 0.1582 (0.0046) | 0.6483 (0.0054) | 0.2754 (0.0003) |
| 10 | 0.2135 (0.0028) | 0.7326 (0.0012) | 0.2576 (0.0003) |
| 17 (square root of 300) | 0.2211 (0.0024) | 0.7400 (0.0016) | 0.2542 (0.0004) |
| 30 | 0.2254 (0.0024) | 0.7432 (0.0010) | 0.2518 (0.0002) |
| 100 | 0.2244 (0.0034) | 0.7411 (0.0016) | 0.2510 (0.0004) |
| 300 (every column, plain bagging) | 0.2208 (0.0042) | 0.7359 (0.0024) | 0.2523 (0.0007) |

| Trees | Validation PR-AUC | Validation ROC-AUC | Validation log loss |
|---:|---:|---:|---:|
| 1 | 0.1368 (0.0049) | 0.6316 (0.0040) | 0.7909 (0.0137) |
| 10 | 0.2052 (0.0002) | 0.7206 (0.0028) | 0.2576 (0.0010) |
| 50 | 0.2222 (0.0030) | 0.7402 (0.0015) | 0.2523 (0.0003) |
| 100 | 0.2254 (0.0024) | 0.7432 (0.0010) | 0.2518 (0.0002) |
| 200 | 0.2266 (0.0026) | 0.7449 (0.0010) | 0.2515 (0.0003) |
| 500 | 0.2274 (0.0026) | 0.7462 (0.0012) | 0.2513 (0.0003) |

![Validation PR-AUC against leaf size, columns per split and number of trees](docs/img/forest.png)

- With one tree the forest scores 0.1368. Averaging 500 of the same trees gives 0.2274, and the validation score did not fall at any tree count up to 500. It is flat from about 200 trees (0.2266 against 0.2274).
- The trees need a minimum leaf size. With `min_samples_leaf` = 1 the training PR-AUC is 1.0 and the validation PR-AUC 0.1967. Averaging many such trees helps, but not enough to make up for the leaves with one applicant each.
- Trying one column per split is the worst setting (0.1582). The score is flat from 30 to 100 columns and drops slightly for all 300 (0.2208), where every tree sees the same strongest columns and the trees are more alike.
- The forest with 500 trees, 30 columns per split and `min_samples_leaf` = 30 has PR-AUC 0.2274 (0.0026), ROC-AUC 0.7462 and log loss 0.2513. The logistic regression has 0.2249 (0.0045), 0.7475 and 0.2495. The PR-AUC difference is smaller than the fold-to-fold std, and the logistic regression is ahead on ROC-AUC and log loss.
- The three settings were chosen on the same folds that are reported, so these scores are slightly optimistic. The holdout is not used yet.

## Gradient boosting with LightGBM

LightGBM fits trees one after another. Each new tree is fitted to the gradient of the log loss at the current predictions, and its output times the learning rate is added to the running score. Settings are LightGBM's defaults (31 leaves per tree, at least 20 applicants per leaf, no row or column sampling, no penalty) except the learning rate, and the pipeline is the same as for the forest. The number of rounds is chosen by early stopping: 10% of each training fold is held out, training stops when its log loss has not improved for 100 rounds, and the model keeps the rounds up to the best one. The validation fold is only used to score the result.

5-fold stratified cross-validation, validation fold, mean (std over folds):

| Learning rate | Best round | Fit time | PR-AUC | ROC-AUC | Log loss |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 21 (4) | 1.7 s | 0.2269 (0.0057) | 0.7489 (0.0022) | 0.2494 (0.0008) |
| 0.10 | 85 (18) | 2.3 s | 0.2366 (0.0063) | 0.7537 (0.0017) | 0.2475 (0.0006) |
| 0.05 | 232 (67) | 3.8 s | 0.2389 (0.0051) | 0.7554 (0.0017) | 0.2471 (0.0006) |
| 0.02 | 581 (108) | 6.9 s | 0.2401 (0.0052) | 0.7563 (0.0017) | 0.2468 (0.0005) |

Fit time is the early-stopped fit on 16 cores, including the 100 rounds after the best one.

![Validation PR-AUC by boosting round for four learning rates, and training against validation log loss](docs/img/lightgbm_rounds.png)

- A smaller learning rate needs more rounds and scores better: PR-AUC 0.2269 at 0.3 and 0.2401 at 0.02. The gain from 0.05 to 0.02 (0.0012) is smaller than the std over folds.
- The validation score is flat for a long time after its best round and does not fall steeply as the decision tree did. At learning rate 0.02 the validation log loss is 0.2468 from round 500 to round 1000 while the training log loss falls from 0.2305 to 0.2182.
- At learning rate 0.3 or 0.1 the validation PR-AUC does turn down after the best round (0.2262 at 20 rounds, 0.2198 at 50 for 0.3). The best round also varies between folds, from 18 to 27 at 0.3 and from 424 to 690 at 0.02.
- LightGBM at 0.02 has PR-AUC 0.2401 (0.0052), against 0.2274 (0.0026) for the forest and 0.2249 (0.0045) for the logistic regression, and ROC-AUC 0.7563 against 0.7462 and 0.7475. The settings other than the learning rate are untuned.
- The score-by-round curves show the mean of the 5 folds at a grid of round numbers, and only the rounds that every fold reached, up to 3 times the fold's best round. The dots are the exact early-stopped results, so they can sit slightly above the curve.

### XGBoost

XGBoost with its defaults (depth-wise trees of at most 6 levels, so at most 64 leaves, histogram split search, no sampling) and the same protocol: same folds, same 10% early-stopping part, log loss with a patience of 100 rounds, same four learning rates. Validation fold, mean (std over folds), with LightGBM from above:

| Learning rate | Library | Best round | Fit time | PR-AUC | ROC-AUC | Log loss |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | LightGBM | 85 (18) | 2.3 s | 0.2366 (0.0063) | 0.7537 (0.0017) | 0.2475 (0.0006) |
| 0.10 | XGBoost | 146 (28) | 5.8 s | 0.2376 (0.0048) | 0.7549 (0.0005) | 0.2472 (0.0004) |
| 0.02 | LightGBM | 581 (108) | 6.9 s | 0.2401 (0.0052) | 0.7563 (0.0017) | 0.2468 (0.0005) |
| 0.02 | XGBoost | 724 (78) | 18.4 s | 0.2401 (0.0050) | 0.7563 (0.0014) | 0.2467 (0.0004) |

- The two libraries give the same scores. Fold by fold, the LightGBM minus XGBoost difference in PR-AUC at learning rate 0.02 has mean 0.0000 and std 0.0012, and at 0.05 it is -0.0005 (std 0.0015).
- XGBoost needs more rounds (724 against 581 at 0.02) and 2.1 to 2.7 times the fit time (LightGBM's time is 0.37 to 0.47 of XGBoost's at each learning rate), on 16 cores.
- Both libraries gain from a smaller learning rate in the same way, from 0.2281 (XGBoost) and 0.2269 (LightGBM) at 0.3 to 0.2401 for both at 0.02.

### Tuning with Optuna

LightGBM at learning rate 0.05 was tuned with Optuna (TPE sampler, seed 42) for 60 trials. Each trial is scored as the mean validation PR-AUC over the same 5 folds, with the rounds chosen by early stopping as above, and a median pruner stops a trial after a fold if its running mean is below the median of earlier trials (never in the first 10 trials). The first trial is the default setting. Every trial is stored with its settings, scores, best round and time in `results/optuna_trials.csv`.

| Setting | Range searched | Default (trial 0) | Best trial (57) |
|---|---|---:|---:|
| `num_leaves` | 8 to 256, log | 31 | 9 |
| `min_child_samples` | 10 to 300, log | 20 | 206 |
| `subsample` (row fraction, every round) | 0.5 to 1 | 1 | 0.85 |
| `colsample_bytree` | 0.3 to 1 | 1 | 0.56 |
| `reg_alpha` (L1 on leaf values) | 0.001 to 10, log | 0.001 | 0.024 |
| `reg_lambda` (L2 on leaf values) | 0.001 to 10, log | 0.001 | 7.5 |

![Optuna history of 60 trials and the importance of each setting](docs/img/optuna_history.png)

- 41 trials completed and 19 were pruned. The study took 24 minutes. The best trial reached a mean PR-AUC of 0.2432, against 0.2385 for the default setting, and 39 of the 41 completed trials beat the default.
- The scores of the best trials are very close: the five best are between 0.2431 and 0.2432, with settings that differ a lot (`num_leaves` from 8 to 27). The surface is flat near the top.
- The best settings are simpler trees (9 leaves) with strong L2 and L1 penalties and sampling of rows and columns. `reg_lambda` and `reg_alpha` explain the most of the variance between trials (0.28 and 0.26 by fANOVA), the sampling fractions 0.19 each, `num_leaves` and `min_child_samples` 0.04 each.
- Several of the best settings sit at the edge of the range searched: `num_leaves` 8 to 9 (lower limit 8) and `reg_lambda` up to 9.7 (upper limit 10). A wider range might find slightly more.

The best trial's score is the best of 60, so it is checked on a new split: the default and the tuned settings were scored on 5 folds drawn with a different seed (7), at learning rate 0.05 and at 0.02. Mean (std over folds), and the mean of the fold-by-fold difference (tuned minus default):

| Learning rate | Settings | Best round | PR-AUC | ROC-AUC | Log loss |
|---:|---|---:|---:|---:|---:|
| 0.05 | default | 226 (51) | 0.2388 (0.0030) | 0.7554 (0.0033) | 0.2471 (0.0005) |
| 0.05 | tuned | 797 (214) | 0.2431 (0.0030) | 0.7587 (0.0037) | 0.2461 (0.0007) |
| 0.05 | difference | | +0.0042 (0.0011) | +0.0033 (0.0014) | -0.0010 (0.0004) |
| 0.02 | default | 643 (151) | 0.2401 (0.0033) | 0.7560 (0.0033) | 0.2469 (0.0006) |
| 0.02 | tuned | 1,937 (551) | 0.2432 (0.0038) | 0.7591 (0.0031) | 0.2460 (0.0006) |
| 0.02 | difference | | +0.0030 (0.0015) | +0.0031 (0.0010) | -0.0009 (0.0003) |

- On the new folds the tuned settings score 0.2431, the same as on the tuning folds (0.2432), so the choice of settings did not overfit those folds.
- The tuned model beats the default in every one of the 10 fold comparisons on all three metrics (smallest PR-AUC gain 0.0012). The gain is small compared with the spread over folds, but it is paired, and it is 0.003 to 0.004 PR-AUC.
- The tuned model needs 3 to 4 times as many rounds, because trees of 9 leaves with strong penalties take smaller steps.
- The holdout is not used yet.
