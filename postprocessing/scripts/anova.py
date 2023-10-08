import numpy as np
import pandas as pd
from statsmodels.api import stats
from statsmodels.formula.api import ols


class AnovaES:
    """
    Add description
    """
    def __init__(self, data: pd.DataFrame, group_contrast: str, covariates: list, ss_type: int = 3):
        self.data = data
        self.ss_type = ss_type
        self.group_contrast = group_contrast
        formula = '{{}} ~ {}'.format(group_contrast)
        for cov in covariates:
            formula += ' + {}'.format(cov)
        self.formula_template = formula
        self.formula_template_only_covariates = '{} ~ '
        for i, cov in enumerate(covariates):
            if i == 0:
                self.formula_template_only_covariates += '{}'.format(cov)
            else:
                self.formula_template_only_covariates += ' + {}'.format(cov)
        self.formula = None

    def anova_es(self, x: str):
        formula = self.formula_template.format(x)
        data = self.data.copy()
        data = data.dropna(subset=[x])
        return self._anova_es(data, formula, self.ss_type)

    @staticmethod
    def _anova_es(data, formula, ss_type):
        """
        Run anova-like linear model with bootstrapped effect size (partial eta2)
        :return:
        """
        # Fit using statsmodels
        lm = ols(formula, data=data).fit()
        aov = stats.anova_lm(lm, typ=ss_type)

        aov = aov.reset_index()
        aov = aov.rename(columns={'index': 'Source',
                                  'sum_sq': 'SS',
                                  'df': 'DF',
                                  'PR(>F)': 'p-unc'})
        aov.index = aov['Source']

        aov['MS'] = aov['SS'] / aov['DF']
        # calculate (partial) eta squared
        aov['n2'] = aov['SS'] / np.sum(aov['SS'])
        aov['np2'] = (aov['F'] * aov['DF']) / (aov['F'] * aov['DF'] + aov.loc['Residual', 'DF'])
        # another way of calculating np2 is:
        # aov['np2'] = aov['SS'] / (aov['SS'] + aov['SS']['Residual']
        # this produces exactly the same partial eta squared values
        return aov

