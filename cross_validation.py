'''
    Cross validation algorithms
'''
import copy
import random

def kfold_items(items, k_folds=10, shuffle=False, seed=None):
    '''
        Splits the data into k folds

        Arguments:
            items : <list>
                The data to be split in k folds
            k_folds : <int>
                The number of folds
                Default value is 10
            shuffle : <bool>
                Whether to shuffle the data before spliting it
                Default value is: False
            seed : <int>
                Sets the seed of the random number generator
                Default value is: None
        Returns:
            split_data : <list>
                The data split in k folds

    '''
    cp_items = copy.deepcopy(items)
    if shuffle:
        tmp_random = random.Random(seed)
        cp_items = tmp_random.sample(cp_items, len(cp_items))

    split_data = []
    num_items = len(cp_items)

    fold_size = num_items // k_folds
    remainder = num_items % k_folds

    start, stop = 0, 0
    for i in xrange(k_folds):
        start = stop
        if i < remainder:
            stop = start + fold_size + 1
        else:
            stop = start + fold_size

        split_data.append(cp_items[start:stop])

    return split_data

def kfold_indices(num_items, k_folds=10, shuffle=False, seed=None):
    '''
        Generates indices for spliting the data into train and test set based on kfolds

        Arguments:
            num_items : <int>
                Number of elements
            k_folds : <int>
                The number of folds
                Default value is: 10
            shuffle : <bool>
                Whether to shuffle or not the data
                Default value is: False
            seed : <int>
                Sets the seed of the random number generator
                Default value is: None
        Returns:
            train_test_indices : <list>
                A list of tuples with lists, where every tuple has indices for train and test set
                (train_indices, test_indices)
    '''
    train_test_indices = []

    sample_indices = range(num_items)
    if shuffle:
        tmp_random = random.Random(seed)
        tmp_random.shuffle(sample_indices)

    fold_size = num_items // k_folds
    remainder = num_items % k_folds

    start, stop = 0, 0
    for i in xrange(k_folds):
        start = stop
        if i < remainder:
            stop = start + fold_size + 1
        else:
            stop = start + fold_size

        test_indices = sample_indices[start:stop]
        train_indices = sample_indices[0:start] + sample_indices[stop:num_items]

        train_test_indices.append((train_indices, test_indices))

    return train_test_indices

def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    len_data = 100

    for train_indices, test_indices in kfold_indices(len_data, 11):
        print train_indices
        print test_indices

if __name__ == '__main__':
    main()
