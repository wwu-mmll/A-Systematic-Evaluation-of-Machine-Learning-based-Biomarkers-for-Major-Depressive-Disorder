import os

from photonai.base import Hyperpipe, PipelineElement, Switch, Stack, Branch, DataFilter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from pandas import Categorical

from macs_datahub.pipeline.pipeline_types.base import PipelineBase, memory


@memory.cache
def cached_rbf(X, Y):
    return rbf_kernel(X, Y)


@memory.cache
def cached_linear(X, Y):
    return linear_kernel(X, Y)


@memory.cache
def cached_poly(X, Y):
    return polynomial_kernel(X, Y)


class AdvancedBase(PipelineBase):

    def create_hyperpipe(self, name):
        pipe = Hyperpipe(name,
                         project_folder=self.path_to_project,
                         optimizer='grid_search',
                         metrics=['balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'accuracy', 'auc'],
                         best_config_metric='balanced_accuracy',
                         outer_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                         inner_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                         nr_of_processes=self.n_processes,
                         cache_folder=os.path.join(self.path_to_project, 'cache'),
                         verbosity=1
                         )
        return pipe

    def _fit(self):
        # fit hyperpipe
        self.pipe.fit(self.X, self.y)

    def add_preproc(self, pipe):
        # add imputer and scaler
        pipe += PipelineElement('SimpleImputer')
        pipe += PipelineElement('RobustScaler')
        return pipe

    def add_imbalanced_data_transform(self, pipe):
        pipe += PipelineElement('ImbalancedDataTransformer', hyperparameters={'method_name': Categorical(['SMOTE'])},
                                test_disabled=False)
        return pipe

    def add_feature_engineering(self, pipe):
        transformer_switch = Switch('TransformerSwitch')
        transformer_switch += PipelineElement('PCA', hyperparameters={'n_components': None}, test_disabled=True)
        transformer_switch += PipelineElement('FClassifSelectPercentile', hyperparameters={'percentile': [5, 10, 50]},
                                              test_disabled=True)
        pipe += transformer_switch
        return pipe

    def add_multi_modal_pca(self, pipe):
        # get modality feature amount to be able to distinguish the features of the different modalities in X since
        # X simply contains modality data as concatenated data in one dimension. Therefore the feature amounts give
        # us the indices.
        output_config = self.config["output_config"]
        output_dir = output_config["output_dir"]
        calc_name = output_config["name"]

        feature_directory = os.path.join(output_dir, calc_name, self.filter,
                                         "data_information/modality_feature_amount.txt")
        with open(feature_directory) as f:
            lines = f.readlines()
        f.close()

        modality_feature_dict = dict()
        for line in lines:
            modality = line.split(":")[0]
            feature_amount = line.split(" ")[1]
            modality_feature_dict[modality] = feature_amount

        # add modalities used to modality_branch
        modality_branches = list()
        feature_range_max = 0
        for modality in modality_feature_dict:
            feature_range = int(modality_feature_dict[modality])
            feature_range_min = feature_range_max
            feature_range_max = feature_range_max + feature_range

            modality_branch = Branch(modality + "_branch")
            modality_branch += DataFilter(indices=range(feature_range_min, feature_range_max))

            modality_branch += PipelineElement('PCA', n_components=None)

            modality_branches.append(modality_branch)

        # create stack and fit
        pipe += Stack('ModalityStack', modality_branches)
        return pipe

    def add_pipeline_elements(self, pipe):
        # add imputer and scaler
        pipe = self.add_preproc(pipe)

        # add multi modal PCA
        if self.is_multi_modal:
            pipe = self.add_multi_modal_pca(pipe)

        # add imbalanced data transformer if set to True by user
        if self.is_imbalanced_data:
            pipe = self.add_imbalanced_data_transform(pipe)

        # add transformer elements
        pipe = self.add_feature_engineering(pipe)
        return pipe

    @property
    def svm_c_hyperparameter(self):
        return [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8]

    def svm_linear(self):
        return PipelineElement('SVC', hyperparameters={'C': self.svm_c_hyperparameter},
                               kernel=cached_linear, max_iter=1000)

    def svm_rbf(self):
        return PipelineElement('SVC', hyperparameters={'C': self.svm_c_hyperparameter},
                               kernel=cached_rbf, max_iter=1000)

    def svm_poly(self):
        return PipelineElement('SVC', hyperparameters={'C': self.svm_c_hyperparameter},
                               kernel=cached_poly, max_iter=1000)

    def random_forest(self):
        return PipelineElement('RandomForestClassifier', hyperparameters={"max_features": ["sqrt", "log2"],
                                                                          "min_samples_leaf": [0.01, 0.1, 0.2]})

    def ada_boost(self):
        return PipelineElement('AdaBoostClassifier', hyperparameters={'n_estimators': [10, 25, 50]})

    def logistic_regression_elastic_net(self):
        return PipelineElement('LogisticRegression',
                                            hyperparameters={"C": [1e-4, 1e-2, 1, 1e2, 1e4],
                                                             "penalty": ['elasticnet']},
                                            l1_ratio=0.5,
                                            solver='saga', n_jobs=1)

    def logistic_regression_l1_l2(self):
        return PipelineElement('LogisticRegression',
                                            hyperparameters={"C": [1e-4, 1e-2, 1, 1e2, 1e4],
                                                             "penalty": ['l1', 'l2']},
                                            solver='saga', n_jobs=1)

    def naive_bayes(self):
        return PipelineElement('GaussianNB')

    def knn(self):
        return PipelineElement('KNeighborsClassifier', hyperparameters={"n_neighbors": [5, 10, 15]})

    def add_estimator(self, pipe):
        estimator_switch = Switch('EstimatorSwitch')

        # SVM
        estimator_switch += self.svm_linear()
        estimator_switch += self.svm_poly()
        estimator_switch += self.svm_rbf()

        # Random Forest
        estimator_switch += self.random_forest()

        # Boosting
        estimator_switch += self.ada_boost()

        estimator_switch += self.logistic_regression_l1_l2()

        estimator_switch += self.logistic_regression_elastic_net()

        estimator_switch += self.naive_bayes()

        estimator_switch += self.knn()

        pipe += estimator_switch
        return pipe


class AdvancedEstimator(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """
        # define hyperpipe
        pipe = self.create_hyperpipe(name='advanced_estimator_pipe')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        pipe = self.add_estimator(pipe)

        self.pipe = pipe


class SVMPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """
        # define hyperpipe
        pipe = self.create_hyperpipe(name='svm_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        switch = Switch('EstimatorSwitch')
        switch += self.svm_linear()
        switch += self.svm_poly()
        switch += self.svm_rbf()

        pipe += switch

        self.pipe = pipe


class RandomForestPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='random_forest_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        pipe += self.random_forest()

        self.pipe = pipe


class BoostingPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='boosting_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        pipe += self.ada_boost()

        self.pipe = pipe


class LogisticRegressionPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='logistic_regression_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        switch = Switch('EstimatorSwitch')
        switch += self.logistic_regression_l1_l2()
        switch += self.logistic_regression_elastic_net()

        pipe += switch

        self.pipe = pipe


class NaiveBayesPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='naive_bayes_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        pipe += self.naive_bayes()

        self.pipe = pipe


class KNNPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='knn_pipeline')

        # add preprocessing, imbalanced data wrapper, multi modal PCA
        pipe = self.add_pipeline_elements(pipe)

        # add estimator elements
        pipe += self.knn()

        self.pipe = pipe


class UnivariateLogisticRegressionPipeline(AdvancedBase):

    def _create(self):
        """ runs the photonai hyperpipe """

        # define hyperpipe
        pipe = self.create_hyperpipe(name='univariate_logistic_regression')

        pipe = self.add_preproc(pipe)

        pipe += PipelineElement('LogisticRegression', penalty='none', n_jobs=1)

        self.pipe = pipe
