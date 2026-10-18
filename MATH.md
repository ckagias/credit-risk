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
