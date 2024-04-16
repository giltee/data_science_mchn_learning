## Seaborn
- statistical graph software

```
conda install seaborn
```

## Seaborn datasets
- comes with their own datasets to test
- tips is the one we will use first

## Distribution Plot

- distplot is deprecated use displot (not noted in the book)
- options:
    - kde: boolean 
    - bins: number

## jointplot

jointplot() allows you to basically match up two distplots for bivariate data. With your choice of what **kind** parameter to compare with: 
* “scatter” 
* “reg” 
* “resid” 
* “kde” 
* “hex”

## corr() function in pandas needs param numeric_only=True