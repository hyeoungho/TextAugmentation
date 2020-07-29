# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:40:09 2020

@author: hybae
"""

import os, sys, logging, traceback, pickle
import collections
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
from sklearn.model_selection import train_test_split

##use downloaded model, change path accordingly
BERT_VOCAB= r'.\BertPretrainedModel\cased_L-12_H-768_A-12\vocab.txt'
BERT_INIT_CHKPNT = r'.\BertPretrainedModel\cased_L-12_H-768_A-12\bert_model.ckpt'
BERT_CONFIG = r'.\BertPretrainedModel\cased_L-12_H-768_A-12\bert_config.json'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids,
        self.is_real_example=is_real_example

def create_examples(df, labels_available, labelcount):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid = row[0]
        text_a = row[1]
        if labels_available:
            labels = row[2:]
        else:
            labels = [0]*labelcount
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples

def convert_examples_to_features(examples,  max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        labels_ids = []
        for label in example.labels:
            labels_ids.append(int(label))

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))
    return features


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
    
    
def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    labels_ids = []
    for label in example.labels:
        labels_ids.append(int(label))


    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=labels_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        #if ex_index % 10000 == 0:
            #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, label_count):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_count], tf.int64),#"label_ids": tf.FixedLenFeature([6], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    print("**********Create_Model*************")
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)#### multi-label case
        
        print(str(labels))
        labels = tf.cast(labels, tf.float32)        
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        print("********model_fn********")
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        print(str(label_ids))
        
        is_real_example = None
        if "is_real_example" in features:
             is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
             is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

                ## original eval metrics
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_loss": loss,
                # }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            print("mode:", mode,"probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn
    
def convertresult(data, thval):
    '''
    # Converts the result to column names based on the threshold value (thval)
    data: input data consist of possible output choices
    thval: threshold value to decide predicted value
    '''
    out = []
    for i in range(0, len(data)):
        pred = data.loc[i, ][data.loc[i, ] > thval].index
        predout = ';'.join(pred)
        out.append(predout)
    out = pd.DataFrame(out, columns = ['Prediction'])
    return(out)

def measureaccuracy(tru, pred, thval):
    '''
    # Compare the groud truth (tru) with the prediction (pred) based upon 
    the threshold value (thval)
    '''
    if len(tru) != len(pred) or all(tru.columns == pred.columns) != True:
        raise Exception("Ground truth should have identical structure as the \
                        prediction result")
    comp = pd.DataFrame(columns=['GroundTruth', 'Predicted', 'Matched'])
    matchcount = 0
    for i in range(0, len(tru)):
        gt = tru.loc[i, ][tru.loc[i, ] > thval].index
        pr = pred.loc[i, ][pred.loc[i, ] > thval].index
        
        matched = True
        for i in range(0, len(gt)):
            if gt[i] not in pr:
                matched = False
                break
        
        if matched:
            comp = comp.append({'GroundTruth':';'.join(gt),'Predicted':';'.join(pr),'Matched':'True'}, ignore_index = True)
            matchcount = matchcount + 1
        else:
            comp = comp.append({'GroundTruth':';'.join(gt),'Predicted':';'.join(pr),'Matched':'False'}, ignore_index = True)
    print("Accuracy is " + str(matchcount/len(tru)))
    return(comp)

if __name__ == "__main__":
    '''
    Usage: python UIFClassifier_Bert_Predict.py Debug inputDataPath modelCheckPointPath variablePath outputDir
    #Expected input
     Debug: True - Load default value 
     inputDataPath: File path to the inputData
     modelCheckPointPath: File path to the checkpoint
     variablePath: File path to the pickle dump file
     outputDir: Directory path to the prediction output     
    '''
    try:
        debug = sys.argv[1]
    except BaseException as e:
        print("Could not load debug argument. In debug mode: " + str(e))
        debug = True
    
    if debug == True:
        inputDataPath = r'.\Data\UIFTestData_BertCompatible.csv'
        #This checkpoint path should be updated based upon the latest training result
        modelCheckPointPath = r'.\Train\Working\checkpoints\model.ckpt-232'
        variablePath = r'.\Train\Working\variable.pckl'
        checkPointBase, _ = os.path.split(modelCheckPointPath)
        outputDir = r'.\Train\Working\predict'
    else:
        if len(sys.argv) != 6:
            raise Exception("Usage: python UIFClassifier_Bert_Predict.py Debug inputDataPath modelCheckPointPath variablePath outputDir")
        inputDataPath = sys.argv[2]
        modelCheckPointPath = sys.argv[3]
        checkPointBase, _ = os.path.split(modelCheckPointPath)
        variablePath = sys.argv[4]
        outputDir = sys.argv[5]
    
    #Load Variable
    try:
        #LABEL_COLUMNS needs to be changed according to the input data
        f = open(variablePath, 'rb')
        [MAX_SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_TRAIN_EPOCHS, SAVE_CHECKPOINTS_STEPS, DO_LOWER_CASE, ID, DATA_COLUMN, LABEL_COLUMNS] = pickle.load(f)
        f.close()
        print("Variables are loaded")
    except BaseException as e:
        print("Loading variable failed:" + str(e))
        print("*** print_tb:")
        traceback.print_exc(file=sys.stdout)
        exit()
    
    ##load inputdata
    try:
        testdata = pd.read_csv(inputDataPath, encoding='latin1')
        columns = testdata.columns
        num_actual_predict_examples = len(testdata)
        if ID not in columns or DATA_COLUMN not in columns:
            raise Exception("InputData does not follow the required format. Id or Text missing")
        
        labelcount = len(LABEL_COLUMNS)
        test_examples = create_examples(testdata, False, labelcount)
        print("Inputdata is loaded")
    except BaseException as e:
        print("Loading inputdta failed:" + str(e))
        print("*** print_tb:")
        traceback.print_exc(file=sys.stdout)
        exit()
    
    #Prepare for prediction
    num_train_steps = int(len(test_examples) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    #num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    test_file = os.path.join(outputDir, "predict.tf_record")
    try:
        if not os.path.exists(test_file):
            open(test_file, 'w').close()
        tokenization.validate_case_matches_checkpoint(DO_LOWER_CASE, modelCheckPointPath)
        tokenizer = tokenization.FullTokenizer(
              vocab_file=BERT_VOCAB, do_lower_case=DO_LOWER_CASE)
        
        file_based_convert_examples_to_features(
                    test_examples, MAX_SEQ_LENGTH, tokenizer, test_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(test_examples))
        tf.logging.info("  Batch size = %d", BATCH_SIZE)
        tf.logging.info("  Num steps = %d", num_train_steps)
    
        test_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False,
            label_count = labelcount)
        
        run_config = tf.estimator.RunConfig(
            #model_dir=outputDir,
            model_dir=checkPointBase,
            #save_summary_steps=SAVE_SUMMARY_STEPS,
            #keep_checkpoint_max=1,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

        bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
        print("Prediction prep is ready")    
    except BaseException as e:
        print("Prediction prep failed:" + str(e))
        print("*** print_tb:")
        traceback.print_exc(file=sys.stdout)
        exit()

    #Prediction
    try:
        model_fn = model_fn_builder(
          bert_config=bert_config,
          num_labels= len(LABEL_COLUMNS),
          init_checkpoint = modelCheckPointPath,
          learning_rate=LEARNING_RATE,
          num_train_steps=None,
          num_warmup_steps=None,
          use_tpu=False,
          use_one_hot_embeddings=False)

        estimator = tf.estimator.Estimator(
          model_fn=model_fn,
          config=run_config,
          params={"batch_size": BATCH_SIZE})
        
        result = estimator.predict(input_fn=test_input_fn)
        
        output_predict_file = os.path.join(outputDir, "test_results.csv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
          num_written_lines = 0
          #Column names
          output_line = ','.join(LABEL_COLUMNS) + "\n"
          writer.write(output_line)
          tf.logging.info("***** Predict results *****")
          for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
              break
            output_line = ",".join(
                str(class_probability)
                for class_probability in probabilities) + "\n"
            writer.write(output_line)
            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        
        print("Prediction completed successfully")
    except BaseException as e:
        print("Prediction failed:" + str(e))
        print("*** print_tb:")
        traceback.print_exc(file=sys.stdout)
        exit()