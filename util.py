'''
    This module holds all the helper methods needed in the other modules.
'''
import os
import re
import inspect
import logging
import multiprocessing

import nltk
import gensim

CLASS_MAP = {
    'MAKEDONIJA': 0,
    'SVET': 1,
    'EKONOMIJA': 2,
    'SCENA': 3,
    'ZDRAVJE': 4,
    'KULTURA': 5,
    'TEHNOLOGIJA': 6,
    'ZIVOT': 7,
    'FUDBAL': 8,
    'KOSARKA': 9,
    'RAKOMET': 10,
    'TENIS': 11,
    'MOTO': 12
}

INVERSE_CLASS_MAP = {
    0: 'MAKEDONIJA',
    1: 'SVET',
    2: 'EKONOMIJA',
    3: 'SCENA',
    4: 'ZDRAVJE',
    5: 'KULTURA',
    6: 'TEHNOLOGIJA',
    7: 'ZIVOT',
    8: 'FUDBAL',
    9: 'KOSARKA',
    10: 'RAKOMET',
    11: 'TENIS',
    12: 'MOTO',
}

def to_unicode(text, code='utf-8'):
    '''
        Converts text into unicode

        Arguments:
            text : <str>
                The text to be converted in unicode
            code : <str>
                Codec value
                Default value is: utf-8
        Returns:
            text : <unicode>
                The converted text
    '''
    if isinstance(text, str):
        if not isinstance(text, unicode):
            text = unicode(text, code)

    return text

def to_utf(text, code='utf-8'):
    '''
        Converts text into bytestring utf-8

        Arguments:
            text : <unicode>
                The text to be converted in bytestring
            code : <str>
                Codec value
                Default value is: utf-8
        Returns:
            text : <str>
                The converted text
    '''
    if isinstance(text, unicode):
        text = text.encode(code)
    elif isinstance(text, str):
        text = unicode(text, code).encode(code)

    return text

def setup_logger(logger_name, log_file, level=logging.INFO, to_stream=True, to_file=True):
    '''
        Creates a logger with the logger_name

        Arguments:
            logger_name : <str>
                The name of the logger that will be created
            log_file : <str>
                The name of the file of where the logs will be writter
            level : logging.LEVEL
                Logging level
    '''
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s',\
        datefmt='%m/%d/%Y %H:%M:%S')

    logger.setLevel(level)

    if to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    if to_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

def get_logger(logger_name):
    '''
        Returns the logger with name logger_name

        Arguments:
            logger_name : <str>
                The name of the logger
        Returns:
            logger : <str>
                The logger with the given logger_name
    '''
    return logging.getLogger(logger_name)

def sentence_tokenizer(text):
    '''
        Tokenizes text into sentences using nltk.sent_tokenize tokenizer

        Arguments:
            text : <unicode>
                The text to be tokenized into sentences
        Returns:
            sentences : <list>
                The sentences in the text
    '''
    sentences = nltk.sent_tokenize(text)

    return sentences

def word_tokenizer(text):
    '''
        Tokenizes text into words using regular expression

        Arguments:
            text : <unicode>
                The text to be tokenized into words
        Returns:
            words : <list>
                The words in the text
    '''
    tokenize_re = re.compile(r'(?u)\w+')

    words = tokenize_re.findall(text)

    return words

def process_line(line):
    '''
        Processes one line of the dataset
        The line is in format:
            class_name\tlink_address\ttext\n

        Arguments:
            line : <str>
                The line to be processed
        Returns:
            processed_line : <tuple>
                The processed line in the format:
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
    '''
    line = to_unicode(line)
    (class_name, link_address, text) = line.split('\t')

    sentences = []

    for sentence in sentence_tokenizer(text):
        sentence = sentence.lower()
        sentence.strip()

        sentences.append(word_tokenizer(sentence))

    return (CLASS_MAP[class_name], link_address, sentences)

def phrase_detection(dataset, delimeter='_'):
    '''
        Detects phrases from the dataset.

        Arguments:
            dataset : <list>
                The dataset on which the phrases should be detected.
            delimeter : <str>
                The glue character used to join phrases.
        Returns:
            dataset : <list>
                The dataset where each phrase have been merge into a single word.
                The dataset is in format:
                A list of tuples in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
    '''
    sentences = []
    for line in dataset:
        for sentence in line[2]:
            sentences.append(sentence)

    phrases = gensim.models.phrases.Phrases(sentences)

    for line_num in range(len(dataset)):
        (class_name, link_address, sentences) = dataset[line_num]
        
        sentences = list(phrases[sentences])
        dataset[line_num] = (class_name, link_address, sentences)

    return dataset

def read_dataset_threaded(file_name, processes=4, logger_name=__name__):
    '''
        Reads the dataset from the file with file_name
        It uses #processes to do the job
        The dataset format in the file is:
            class_name\tlink_address\ttext\n

        Arguments:
            file_name : <str>
                The file name of the file to be read
            processes : <int>
                Number of workers to be used
                Default value is: 4
        Returns:
            dataset : <list>
                Returns a list of tuples in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
    '''
    logger = get_logger(logger_name)

    logger.info('Function={0}, FileName={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        file_name,
        'Started processing file'
        ))
    
    file_reader = open(file_name, 'r')
    lines = [line for line in file_reader]
    file_reader.close()

    pool = multiprocessing.Pool(processes=processes)
    dataset = pool.map(process_line, lines)
    pool.close()
    pool.join()

    return dataset

def read_dataset(file_name, logger_name=__name__):
    '''
        Reads the dataset from the file with file_name
        The dataset format in the file is:
            class_name\tlink_address\ttext\n

        Arguments:
            file_name : <str>
                The file name of the file to be read
        Returns:
            dataset : <list>
                Returns a list of tuples in format
                    (class_id <str>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
    '''
    logger = get_logger(logger_name)

    logger.info('Function={0}, FileName={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        file_name,
        'Started processing file'
        ))

    file_reader = open(file_name, 'r')
    lines = [line for line in file_reader]
    file_reader.close()

    dataset = []
    for line in lines:
        line = to_unicode(line)
        (class_name, link_address, text) = line.split('\t')

        sentences = []

        for sentence in sentence_tokenizer(text):
            sentence = sentence.lower()
            sentence.strip()

            sentences.append(word_tokenizer(sentence))

        dataset.append((CLASS_MAP[class_name], link_address, sentences))

    return dataset

def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    logger_name = 'Testing'
    setup_logger(logger_name, 'Test.log', to_file=False)

    dataset = read_dataset_threaded(os.path.join('data', 'raw_texts.txt'), processes=4,
        logger_name=logger_name)

    dataset = phrase_detection(dataset)

    print dataset
    for line in dataset:
        for sentence in line[2]:
            for word in sentence:
                print word,
                print '-',
            print ''

if __name__ == '__main__':
    main()
