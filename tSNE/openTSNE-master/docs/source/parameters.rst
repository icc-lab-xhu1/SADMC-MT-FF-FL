.. _parameter-guide:

Parameter guide
===============


Perplexity
----------
Perplexity is perhaps the most important parameter in t-SNE and can reveal different aspects of the data. Considered loosely, it can be thought of as the balance between preserving the global and the local structure of the data. A more direct way to think about perplexity is that it is the continuous analogy to the :math:`k` number of nearest neighbors for which we will preserve distances.

In most implementations, perplexity defaults to 30. This focuses the attention of t-SNE on preserving the distances to its 30 nearest neighbors and puts virtually no weight on preserving distances to the remaining points. For data sets with a small number of points e.g. 100, this will uncover the global structure quite well since each point will preserve distances to a third of the data set.

For larger data sets, e.g. 10,000 points, considering 30 nearest neighbors will likely do a poor job of preserving global structure. Using a higher perplexity value e.g. 500, will do a much better job for of uncovering the global structure. For larger data sets still e.g. 500k or 1 million samples, this is typically not enough and can take quite a long time to run. Luckily, various tricks can be used to improve global structure [1]_.

.. figure:: images/macosko_perplexity.png

    **Figure 1**: Higher values of perplexity do a better job of preserving global structure, but can obscure local structure. In both a) and b) we run standard t-SNE with perplexities 30 and 500, respectively.

Note that perplexity linearly impacts runtime i.e. higher values of
perplexity will incur longer execution time. For example, the embedding in Figure 1a took around 1 minute 30 seconds to compute, while Figure 1b took around 6 minutes.


Exaggeration
------------

The exaggeration factor is typically used during the early exaggeration phase. This factor increases the attractive forces between points and allows points to move around more freely, finding their nearest neighbors more easily. The most typical value of exaggeration during the early exaggeration phase is 12, but higher values have also been shown to work in combination with different learning rates [2]_.

Exaggeration can also be used during the normal optimization regime to form more densely packed clusters, making the separation between clusters more visible [1]_.

.. figure:: images/10x_exaggeration.png

    **Figure 2**: We run t-SNE twice on the 10x genomics mouse brain data set, containing 1,306,127 samples. a) t-SNE was run with the regular early exaggeration phase 12 for 500 iterations, then in the regular regime with no exaggeration for 750 iterations. b) t-SNE was run  with the regular early exaggeration phase 12 for 500 iterations, then for another 750 iterations with exaggeration 4.

Optimization parameters
-----------------------

t-SNE uses a variation of gradient descent optimization procedure that incorporates momentum to speed up convergence of the embedding [3]_.

learning_rate: float
    The learning rate controls the step size of the gradient updates. This parameter can be manually set, however, we recommend using the default value of "auto", which sets the learning rate by dividing the number of samples by the exaggearation factor.

momentum: float
    To increase convergence speed and reduce the number of iterations required, we can augment gradient descent with a momentum term. Momentum stores an exponentially decaying sum of gradient updates from previous iterations. By default, this is typically set to 0.8.

max_grad_norm: float
    By default, openTSNE does not apply gradient clipping. However, when embedding new data into an existing embedding, care must be taken that the data points do not "shoot off". Gradient clipping alevaites this issue.


Barnes-Hut parameters
---------------------

Please refer to :ref:`barnes-hut` for a description of the Barnes-Hut algorithm.

theta: float
    The trade-off parameter between accuracy and speed.


Interpolation parameters
------------------------

Please refer to :ref:`fit-sne` for a description of the interpolation-based algorithm.

n_interpolation_points: int
    The number of interpolation points to use within each grid cell. It is highly recommended leaving this at the default value due to the Runge phenomenon described above.

min_num_intervals: int
    This value indicates what the minimum number of intervals/cells should be in any dimension.

ints_in_interval: float
    Our implementation dynamically determines the number of cells such that the accuracy for any given interval remains fixed. This value indicates the size of the interval/cell in any dimension e.g. setting this value to 3 indicates that all the cells should have side length of 3.


References
----------
.. [1] Kobak, Dmitry, and Berens, Philipp. `“The art of using t-SNE for single-cell transcriptomics” <https://www.nature.com/articles/s41467-019-13056-x>`__, Nature Communications (2019).

.. [2] Linderman, George C., and Stefan Steinerberger. `“Clustering with t-SNE, provably.” <https://epubs.siam.org/doi/abs/10.1137/18M1216134>`__, SIAM Journal on Mathematics of Data Science (2019).

.. [3] Jacobs, Robert A. `"Increased rates of convergence through learning rate adaptation." <https://www.sciencedirect.com/science/article/abs/pii/0893608088900032>`__, Neural Networks (1988).
