"""A set of helper functions to print out informations"""

__copyright__ = 'Copyright 2017, Instronizer'
__credits__ = ['Michał Martyniak', 'Maciej Rutkowski', 'Filip Schodowski']
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Production'

import time
import datetime
import numpy as np


def print_execution_time(function):
    """
    Decorator which measures function's execution time
    Just add @print_execution_time above your function definition
    """
    def wrapper(*args, **kw):
        start_time = time.clock()
        result = function(*args, **kw)
        formatted_time_took = datetime.timedelta(seconds=(time.clock() - start_time))
        print('Function {} took: {}'.format(
            function.__name__, formatted_time_took))
        return result

    return wrapper

def print_args(args):
    print('=== PARAMETERS ==============================')
    for arg in vars(args):
        print(arg.upper(), '=', getattr(args, arg))
    print('=== END =====================================\n\n')


def print_validation_info(target, output_data, class_names, classes_on_timeline, classes_counter):
    predictions = [np.argmax(output_row.numpy()) for output_row in output_data]
    pairs = zip(list(target), predictions)

    print('Model output: {}'.format(output_data))
    print('Target     | Prediction')
    print('-----------------------')
    for pair in pairs:
        classes_counter[pair[1]] += 1
        classes_on_timeline.append(pair[1])
        print('{:10s} | {:10s}'.format(
            class_names[pair[0]], class_names[pair[1]]))
    print()


def print_test_info(output_data, class_names, classes_on_timeline, classes_counter, current_index, max_index):
    predictions = [np.argmax(output_row.numpy()) for output_row in output_data]

    print('[{}/{}] Model output: {}'.format(current_index+1, max_index, output_data))
    print('Prediction:')
    for predicted_class_index in predictions:
        classes_counter[predicted_class_index] += 1
        classes_on_timeline.append(predicted_class_index)
        print(class_names[predicted_class_index])
    print()


def print_class_counters(classes, predictions_counter):
    results = zip(classes, predictions_counter)
    for class_index, (class_name, counter) in enumerate(results):
        print('({}) {}: {} |'.format(class_index, class_name, counter), end=' ')
    print()
