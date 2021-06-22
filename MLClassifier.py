"""
the only ML module in the solution;
has methods for training, evaluation and inference;
in the ideal world the latter would be separated from the former ones;
relies on BERT and pytorch framework
"""

import fasttext
import os
import pandas as pd
from transformers import BertTokenizer
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import os

# todo: move to config
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class MLClassifier:

    def __init__(self):
        self.dataset = None
        self.device = None

    def load_device(self):
        if torch.cuda.is_available():
            # tell pytorch to use the GPU
            self.device = torch.device("cuda")
            print("There are {} GPU(s) available.".format(torch.cuda.device_count()))
            print("We will use the GPU:{}.".format(torch.cuda.get_device_name(0)))
        else:
            print("No GPU available, using the CPU instead.")
            self.device = torch.device("cpu")

    def load_train_dataset(self):
        df = pd.read_csv("./data/in_domain_train.tsv", delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])

        print('Number of training sentences: {}\n'.format(df.shape[0]))

        # Display 5 random rows from the data.
        print(df.sample(5))

        self.dataset = df

    def tokenize_data(self):
        # load the BERT tokenizer.
        print('loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # tokenize all of the sentences and map the tokens to their word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in self.dataset.sentence.values:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self.dataset.label.values)

        return input_ids, attention_masks, labels, tokenizer

    @staticmethod
    def split_into_train_validation(input_ids, attention_masks, labels):

        # combine the training inputs into a TensorDataset
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # create a 90-10 train-validation split

        # calculate the number of samples to include in each set
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # divide the dataset by randomly selecting samples
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{0} training samples'.format(train_size))
        print('{0} validation samples'.format(val_size))
        return train_dataset, val_dataset

    @staticmethod
    def init_data_loaders(train_data, val_data):

        # The DataLoader needs to know our batch size for training, so we specify it
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch
        # size of 16 or 32.
        batch_size = 32

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_data,  # The training samples.
            sampler=RandomSampler(train_data),  # Select batches randomly
            batch_size=batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_data,  # The validation samples.
            sampler=SequentialSampler(val_data),  # Pull out batches sequentially.
            batch_size=batch_size  # Evaluate with this batch size.
        )

        return train_dataloader, validation_dataloader

    @staticmethod
    def flat_accuracy(predicted, labels):
        predicted_flat = np.argmax(predicted, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(predicted_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        # round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_classifier(self, train_dataloader, validation_dataloader):

        # load BertForSequenceClassification model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # number of training epochs, between 2 and 4 are recommended;
        # 2 is chosen due to time constraints
        epochs = 2

        # total number of training steps is [number of batches] x [number of epochs]
        total_steps = len(train_dataloader) * epochs

        # create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
            # perform one full pass over the training set.

            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # reset the total loss for this epoch.
            total_train_loss = 0

            # put the model into training mode
            model.train()

            # for each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    print('Batch {} out of {}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # unpack the training batch from the dataloader and copy each tensor to the device
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()

                # perform a forward pass (evaluate the model on this training batch)
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
                loss = result.loss
                print(loss)

                # accumulate total loss
                total_train_loss += loss.item()

                # perform a backward pass to calculate the gradients
                loss.backward()

                # clip the norm of the gradients to 1.0
                # this is to help prevent the "exploding gradients" problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # update parameters and take a step using the computed gradient
                optimizer.step()

                # update the learning rate.
                scheduler.step()

            # calculate the average loss over all of the batches
            avg_train_loss = total_train_loss / len(train_dataloader)

            # measure how long this epoch took
            training_time = self.format_time(time.time() - t0)

            print("\nAverage training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}".format(training_time))

            # after the completion of each training epoch, measure our performance on
            # our validation set.
            print("Running Validation...")

            t0 = time.time()

            # put the model in evaluation mode--the dropout layers behave differently
            # during evaluation
            model.eval()

            # tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    result = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

                    loss = result.loss
                    logits = result.logits

                # accumulate the validation loss
                total_eval_loss += loss.item()

                # move logits and labels to CPU (if they were on GPU)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # report the final accuracy for this validation run
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("Accuracy: {0:.2f}".format(avg_val_accuracy))

            # calculate the average loss over all of the batches
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("Validation Loss: {0:.2f}".format(avg_val_loss))
            print("Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))
        return model

    @staticmethod
    def save_model(model, tokenizer):
        output_dir = './models/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    @staticmethod
    def load_tokenizer_and_model(model_folder):
        try:
            tokenizer = BertTokenizer.from_pretrained(model_folder)
            model_loaded = BertForSequenceClassification.from_pretrained(model_folder)
            return tokenizer, model_loaded
        except:
            return None

    @staticmethod
    def load_testdata(tokenizer):
        df = pd.read_csv("./data/out_of_domain_dev.tsv", delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])

        print('Number of initial test sentences: {}\n'.format(df.shape[0]))
        # df = df[:100]
        # print('Number of test sentences: {}\n'.format(df.shape[0]))
        # Create sentence and label lists
        sentences = df.sentence.values
        labels = df.label.values

        # tokenize all of the sentences and map the tokens to their word IDs
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=64,
                pad_to_max_length=True,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )

            # add the encoded sentence to the list
            input_ids.append(encoded_dict['input_ids'])

            # and its attention mask (simply differentiates padding from non-padding)
            attention_masks.append(encoded_dict['attention_mask'])

        # convert the lists into tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # set the batch size
        batch_size = 32

        # create the DataLoader
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        return prediction_dataloader

    def get_predictions(self, model, prediction_dataloader):
        print('Predicting labels for test sentences...')

        # put model in evaluation mode
        model.eval()

        predictions, true_labels = [], []

        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            # unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        return predictions, true_labels

    @staticmethod
    def evaluate_predictions(predicted_labels, true_labels):

        flat_predictions = np.concatenate(predicted_labels, axis=0)

        # for each sample, pick the label (0 or 1) with the higher score
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        # calculate the metrics
        acc = accuracy_score(flat_true_labels, flat_predictions)
        precision = precision_score(flat_true_labels, flat_predictions)
        recall = recall_score(flat_true_labels, flat_predictions)
        f1 = f1_score(flat_true_labels, flat_predictions)
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

        print('ACC: %.3f' % acc)
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1: %.3f' % f1)
        print('MCC: %.3f' % mcc)

        with open('classifier_eval_results.txt', 'w') as f:
            print('ACC: %.3f\n' % acc, file=f)
            print('Precision: %.3f\n' % precision, file=f)
            print('Recall: %.3f\n' % recall, file=f)
            print('F1: %.3f\n' % f1, file=f)
            print('MCC: %.3f\n' % mcc, file=f)

    # obsolete
    @staticmethod
    def test_embeddings():

        data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        model = fasttext.load_model(os.path.join(data_folder_path, 'fasttext_twitter_raw.bin'))

        word_inv = 'running'
        word_oov = 'runnnnnnnnnnnning'

        if word_oov in model:
            print('This word is in the vocabulary: {}'.format(word_oov))
        else:
            print('This word is NOT in the vocabulary: {}'.format(word_oov))

        # print('The vector for the the word {} is:'.format(word_oov))
        # print(model[word_oov])

        print(model.get_nearest_neighbors('lol'))

    # evaluates (classifies) a single sentence by the model
    def evaluate_single_sentence(self, sentence):
        # load the model from the folder
        tokenizer, model = self.load_tokenizer_and_model(MODEL_FOLDER)
        if not tokenizer or not model:
            print("No model to evaluate. Exiting.")
            return -1

        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        # add the encoded sentence to the list
        input_id = encoded_dict['input_ids']

        # and its attention mask (simply differentiates padding from non-padding).
        attention_mask = encoded_dict['attention_mask']
        input_id = torch.LongTensor(input_id)
        attention_mask = torch.LongTensor(attention_mask)
        model_loaded = model.to(self.device)
        input_id = input_id.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            # forward pass, calculate logit predictions
            outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs[0]
        index = logits.argmax()
        return index == 1


# train the model
def run_training():
    cls = MLClassifier()
    cls.load_device()
    cls.load_train_dataset()
    input_ids, attention_masks, labels, tokenizer = cls.tokenize_data()
    train_ds, val_ds = cls.split_into_train_validation(input_ids, attention_masks, labels)
    train_dataloader, val_dataloader = cls.init_data_loaders(train_ds, val_ds)
    model = cls.train_classifier(train_dataloader, val_dataloader)
    cls.save_model(model, tokenizer)


# evaluate the model
def run_evaluation():
    cls = MLClassifier()
    tokenizer, model = cls.load_tokenizer_and_model(MODEL_FOLDER)
    prediction_dataloader = cls.load_testdata(tokenizer)
    predicted_labels, true_labels = cls.get_predictions(model, prediction_dataloader)
    cls.evaluate_predictions(predicted_labels, true_labels)


if __name__ == "__main__":
    # run_training()
    run_evaluation()
