import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators

def time_series_maximum(x):
    """
    序列x的最大值
    :param x: x
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.maximum(x)

def time_series_minimum(x):
    """
    序列x的最小值
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.minimum(x)

def time_series_mean(x):
    """
    序列x的平均值
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean(x)

def time_series_variance(x):
    """
    序列x的方差
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.variance(x)

def time_series_standard_deviation(x):
    """
    序列x的标准差
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.standard_deviation(x)

def time_series_skewness(x):
    """
    序列x的偏度
    数据分布的左偏或右偏，指的是数值拖尾的方向，而不是峰的位置
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.skewness(x)

def time_series_kurtosis(x):
    """
    序列x的峰度
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.kurtosis(x)

def time_series_median(x):
    """
    序列x中位数
    Returns the median of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.median(x)

def time_series_abs_energy(x):
    """
    np.dot()
    序列X的点积
    Returns the absolute energy of the time series which is the sum over the squared values
    .. math::
        E = \\sum_{i=1,\ldots, n} x_i^2
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.abs_energy(x)

def time_series_absolute_sum_of_changes(x):
    """
    一阶差分绝对值之和
    Returns the sum over the absolute value of consecutive changes in the series x
    .. math::
        \\sum_{i=1, \ldots, n-1} \\mid x_{i+1}- x_i \\mid
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.absolute_sum_of_changes(x)

def time_series_variance_larger_than_std(x):
    """
    序列x的方差是否大于标准差，大于则为1
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return int(ts_feature_calculators.variance_larger_than_standard_deviation(x))

def time_series_count_above_mean(x):
    """
    序列x中比mean(x)大的数目
    Returns the number of values in x that are higher than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_above_mean(x)

def time_series_count_below_mean(x):
    """
    序列x中比mean(x)小的数目
    Returns the number of values in x that are lower than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_below_mean(x)

def time_series_first_location_of_maximum(x):
    """
    序列x中最大值首次出现的位置与序列长度的相对值
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_maximum(x)

def time_series_first_location_of_minimum(x):
    """
    序列x中最小值首次出现的位置与序列长度的相对值
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_minimum(x)

def time_series_last_location_of_maximum(x):
    """
    序列x中最大值最后出现的位置与序列长度的相对值
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_maximum(x)

def time_series_last_location_of_minimum(x):
    """
    序列x中最小值最后出现的位置与序列长度的相对值
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_minimum(x)

def time_series_has_duplicate(x):
    """
    序列x是否存在重复值，存在为false，否则为true
    Checks if any value in x occurs more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return ts_feature_calculators.has_duplicate(x)

def time_series_has_duplicate_max(x):
    """
    序列x是否存在重复的最大值
    Checks if the maximum value of x is observed more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return int(ts_feature_calculators.has_duplicate_max(x))

def time_series_has_duplicate_min(x):
    """
    序列x是否存在重复的最小值
    Checks if the minimal value of x is observed more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return int(ts_feature_calculators.has_duplicate_min(x))

def time_series_longest_strike_above_mean(x):
    """
    序列x中大于mean(x)的最长子序列的长度
    np.max(_get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_above_mean(x)

def time_series_longest_strike_below_mean(x):
    """
    序列x中小于mean(x)的最长子序列的长度
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_below_mean(x)

def time_series_mean_abs_change(x):
    """
    序列x的一阶差分的绝对值的平均值
    Returns the mean over the absolute differences between subsequent time series values which is
    .. math::
        \\frac{1}{n} \\sum_{i=1,\ldots, n-1} | x_{i+1} - x_{i}|
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_abs_change(x)

def time_series_mean_change(x):
    """
    序列x的一阶差分的平均值
    Returns the mean over the absolute differences between subsequent time series values which is
    .. math::
        \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_change(x)

def time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    """
    序列x中的重复数据占比
    Returns the percentage of unique values, that are present in the time series
    more than once.
        len(different values occurring more than once) / len(different values)
    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_values_to_all_values.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(x)


def time_series_ratio_value_number_to_time_series_length(x):
    """
    序列x中去重后的数据长度占原序列x的比例
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns
        # unique values / # values
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.ratio_value_number_to_time_series_length(x)


def time_series_sum_of_reoccurring_data_points(x):
    """
    序列x中所有重复数据的和
    Returns the sum of all data points, that are present in the time series
    more than once.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_data_points(x)


def time_series_sum_of_reoccurring_values(x):
    """
    序列x中所有重复数据的和(每个重复数据只加一次)
    Returns the sum of all values, that are present in the time series
    more than once.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_values(x)


def time_series_sum_values(x):
    """
    序列x的和
    Calculates the sum over the time series values
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.sum_values(x)


def time_series_range(x):
    """
    序列x最大值和最小值的差值
    Calculates the range value of the time series x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return time_series_maximum(x) - time_series_minimum(x)

# add yourself statistical features here...


def get_statistical_features(x):
    statistical_features = [
        time_series_maximum(x),
        time_series_minimum(x),
        time_series_mean(x),
        time_series_variance(x),
        time_series_standard_deviation(x),
        time_series_skewness(x),
        time_series_kurtosis(x),
        time_series_median(x),
        time_series_abs_energy(x),
        time_series_absolute_sum_of_changes(x),
        time_series_variance_larger_than_std(x),
        time_series_count_above_mean(x),
        time_series_count_below_mean(x),
        time_series_first_location_of_maximum(x),
        time_series_first_location_of_minimum(x),
        time_series_last_location_of_maximum(x),
        time_series_last_location_of_minimum(x),
        int(time_series_has_duplicate(x)),
        int(time_series_has_duplicate_max(x)),
        int(time_series_has_duplicate_min(x)),
        time_series_longest_strike_above_mean(x),
        time_series_longest_strike_below_mean(x),
        time_series_mean_abs_change(x),
        time_series_mean_change(x),
        time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x),
        time_series_ratio_value_number_to_time_series_length(x),
        time_series_sum_of_reoccurring_data_points(x),
        time_series_sum_of_reoccurring_values(x),
        time_series_sum_values(x),
        time_series_range(x)
    ]
    # append yourself statistical features here...

    return statistical_features