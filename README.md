# transethnic_prs

In this reporsitory, we implement solver to build PRS using penalized regression.

In particular, we focus on the setting where we have following two data sets:

* We have full summary statistics of a large GWAS. 
* We have individual-level genotype and phenotype data for the target population.

The goal is to train a PRS for the target population while leveraging the information from the large GWAS.

We take a penalized regression approach in which we considered two slightly different parameterizations for the task.
And depending on the parameterization, we introduce lasso, group lasso, or sparse group lasso penalty as regularizations.

Detailed notes will come soon! TODO!  

# Important notes

* We assume that the phenotype is standardized. This is required so that the coefficients will always be kept in a fixed scale which does not depend on the phenotype scale. 

<!-- # Running unittest

```
python -m unittest discover -p "*_test.py"
``` -->
