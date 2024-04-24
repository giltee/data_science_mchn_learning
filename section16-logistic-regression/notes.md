# Logistic Regression
- Used on qualitive problems
- We can use a confusion matrix to evaluate our model.
- example, testing for a disease:
| n=165 | Predicted (no) | Predicted (Yes)
| Actual No | 50 | 10 |
| Actual Yes | 5 | 100 |


TP = 150
FP = 10
FN = 5

$$ F1 = {TP \over TP + {1 \over 2 } (FP + FN)} $$

150 / (150 + 1/2(10 + 5))

= 150 / 150 + 7.5
= 150 / 157.5
= 