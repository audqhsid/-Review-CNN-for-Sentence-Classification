import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # re.sub(pattern, repl, string): string에서 pattern과 매치하는 텍스트를 repl로 대체한다
    # regular expression: ^은 not을 의미
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # string에서 [^A-Za-z0-9(), !?\'\`](A to Z, a to z, 0 to 9이 아닌 것들과 !, ?, ', `)를 공백으로 대체한 후 string에 저장
    string = re.sub(r"\'s", " \'s", string) # string에서 's를 's로 대체한 후 string에 저장
    string = re.sub(r"\'ve", " \'ve", string) # string에서 've를 've로 대체한 후 string에 저장
    string = re.sub(r"n\'t", " n\'t", string) # string에서 n't를 n't로 대체한 후 string에 저장
    string = re.sub(r"\'re", " \'re", string) # string에서 're를 're로 대체한 후 string에 저장
    string = re.sub(r"\'d", " \'d", string) # string에서 'd를 'd로 대체한 후 string에 저장
    string = re.sub(r"\'ll", " \'ll", string) # string에서 'll를 'll로 대체한 후 string에 저장
    string = re.sub(r",", " , ", string) # string에서 ,를  " , "로 대체한 후 string에 저장 (앞뒤에 공백을 추가)
    string = re.sub(r"!", " ! ", string) # string에서 !를 " ! "로 대체한 후 string에 저장 (앞뒤에 공백을 추가)
    string = re.sub(r"\(", " \( ", string) # string에서 (를 " ( "로 대체한 후 string에 저장 (앞뒤에 공백을 추가)
    string = re.sub(r"\)", " \) ", string) # string에서 )를 " ) "로 대체한 후 string에 저장 (앞뒤에 공백을 추가)
    string = re.sub(r"\?", " \? ", string) # string에서 ?를 " ? "로 대체한 후 string에 저장 (앞뒤에 공백을 추가)
    string = re.sub(r"\s{2,}", " ", string) # string에서 \s{2,}(\s가 2회 이상인 것)을 공백으로 대체한 후 string에 저장
    # \s: whitespace문자가 아닌 것과 매치 (== [^ \t\n\r\f\v])
    return string.strip().lower() # string의 양쪽에 있는 공백을 지우고 문자들을 소문자로 바꾼 것을 return


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_data_file이라는 파일을 읽기모드, utf-8로 인코딩하여 open, 그 후 파일의 내용을 한꺼번에 한 줄씩 구분하여 읽는다. 이를 각 줄을 list의 항목들로 positive_examples에 저장
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples] # positive_examples에 있는 항목들의 앞 뒤 공백을 제거하여 저장
    # negative_data_file이라는 파일을 읽기모드, utf-8로 인코딩하여 open, 그 후 파일의 내용을 한꺼번에 한 줄씩 구분하여 읽는다. 이를 각 줄을 list의 항목들로 negative_examples에 저장
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples] # negative_examples에 있는 항목들의 앞 뒤 공백을 제거하여 저장
    # Split by words
    x_text = positive_examples + negative_examples # positive_examples 리스트 뒤에 negative_examples 리스트를 추가 (concatenate한 것과 같음)
    x_text = [clean_str(sent) for sent in x_text] # x_text에 있는 항목들을 위에 정의한 clean_str() 함수에 대입한 결과를 저장
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples] # positive_examples에 있는 항목 수 만큼 [0, 1]을 항목으로 하는 positive_labels 생성
    negative_labels = [[1, 0] for _ in negative_examples] # negative_examples에 있는 항목 수 만큼 [1, 0]을 항목으로 하는 negative_labels 생성
    y = np.concatenate([positive_labels, negative_labels], 0) # positive_labels와 negative_labels을 shape(positive_examples에 있는 항목 수 + negative_examples에 있는 항목 수, 2)로 concatenate
    return [x_text, y] # list화 해서 return


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data) # data를 numpy.ndarray로 type casting
    data_size = len(data) # data의 길이를 data_size에 저장
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 # number of batches per epoch 계산
    for epoch in range(num_epochs): # num_epochs만큼 for문 반복
        # Shuffle the data at each epoch
        if shuffle:
            # np.arange: numpy 버전 range
            shuffle_indices = np.random.permutation(np.arange(data_size)) # data_size(0, 1, 2, ..., len(data))를 permuataion(순서를 섞어 나열)하여 shuffle_indices에 저장 (e.g. 19, 2, 0, ..., 1)
            shuffled_data = data[shuffle_indices] # data를 shuffle_indices 순서대로 shuffled_data에 저장
        else:
            shuffled_data = data # 매개변수 shuffle이 'False'이므로 shuffle하지 않은 data를 shuffled_data에 저장
        for batch_num in range(num_batches_per_epoch):# num_batches_per_epoch만큼 for문 반복
            start_index = batch_num * batch_size # start_index 생성
            end_index = min((batch_num + 1) * batch_size, data_size) # end_index 생성
            # yield: 함수가 generator를 반환하는 것 빼고는 return과 비슷하게 사용되는 키워드
            yield shuffled_data[start_index:end_index] # batch_iter 함수 호출 시 for문 한 번 돌 때 마다 shuffled_data의 항목 [start_inex]부터 [end_index-1]까지를 매번 반환
