""" This module prepares midi file data and feeds it to the neural network for training """
import os
import numpy
import pickle
import constants as c
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from music21 import converter, instrument, note, chord, stream


def train_network():
    """ Train a Neural Network to generate new melody """
    keyboard_notes, string_notes = get_all_notes()
    n_vocab_str_notes = len(set(string_notes))

    x = prepare_sequences(keyboard_notes)
    y = prepare_sequences(string_notes)

    model = create_network(x, n_vocab_str_notes)
    train(model, x, y)


def get_all_notes():
    """ Save all notes in songs """
    cmp_k_notes = []
    cmp_s_notes = []

    with open(c.SONG_LIST) as f:
        for line in f:
            midi_file_path = midi_file_path.rstrip()

            cmp_k_notes = get_notes_chords_rests(c.KEYBOARD_INSTRUMENTS, midi_file_path, cmp_k_notes)
            cmp_s_notes = get_notes_chords_rests(c.STRING_INSTRUMENTS, midi_file_path, cmp_s_notes)
            cmp_k_notes, cmp_s_notes = note_sanity_check(cmp_k_notes, cmp_s_notes)

    with open(c.SAVED_KEYB_NOTES, 'wb') as file_path:
        pickle.dump(cmp_k_notes, file_path)
    with open(c.SAVED_STR_NOTES, 'wb') as file_path:
        pickle.dump(cmp_s_notes, file_path)

    return cmp_k_notes, cmp_s_notes


def get_notes_chords_rests(instrument_type, path, note_list):
    """ Get all the notes, chords and rests from the midi files in the song directory """
    try:
        midi_file = converter.parse(path)
        parts = instrument.partitionByInstrument(midi_file)
        for music_instrument in range(len(parts)):
            if parts.parts[music_instrument].id in instrument_type:
                for element_by_offset in stream.iterator.OffsetIterator(parts[music_instrument]):
                    for entry in element_by_offset:
                        if isinstance(entry, note.Note):
                            check_rest_amount(entry, note_list)
                            note_list.append(str(entry.pitch))
                        elif isinstance(entry, chord.Chord):
                            check_rest_amount(entry, note_list)
                            note_list.append('.'.join(str(n) for n in entry.normalOrder))
                        elif isinstance(entry, note.Rest):
                            check_rest_amount(entry, note_list)
                            note_list.append('Rest')
    except Exception as e:
        print("failed on ", path, e)
        pass
    return note_list


def check_rest_amount(element, note_list):
    """Check if there is not a missing rests which ensures silent part in melody"""
    if element.offset / 0.5 <= len(note_list):
        return
    else:
        while True:
            note_list.append('Rest')
            if element.offset / 0.5 <= len(note_list):
                return


def note_sanity_check(keyboard_notes, string_notes):
    """Check note length and normalize it"""
    if len(keyboard_notes) != len(string_notes):
        notes_max = max(len(keyboard_notes), len(string_notes))
        notes_min = min(len(keyboard_notes), len(string_notes))

        if notes_max == len(keyboard_notes):
            del keyboard_notes[notes_min:]
        else:
            del string_notes[notes_min:]

    return keyboard_notes, string_notes


def prepare_sequences(notes):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    network_data = []

    # get all pitch names
    pitch_names = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    # translate notes to their int values
    for i in range(0, len(notes) - sequence_length, 1):
        sequence = notes[i:i + sequence_length]
        network_data.append([note_to_int[char] for char in sequence])

    n_patterns = len(network_data)
    # reshape the input into a format compatible with LSTM layers
    network_data = numpy.reshape(network_data, (n_patterns, sequence_length, 1))
    # normalize input, one-hot-encoding
    network_data = np_utils.to_categorical(network_data)

    return network_data


def create_network(network_input, n_vocab_str_notes):
    """ create the structure of the neural network """
    model = Sequential()

    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dense(256)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(n_vocab_str_notes, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    if os.path.isfile(c.WEIGHTS_PATH):
        print("Resumed model's weights from {}".format(c.WEIGHTS_PATH))
        model.load_weights(c.WEIGHTS_PATH)

    batch_size = 128
    epochs = 200

    checkpoint = ModelCheckpoint(c.WEIGHTS_PATH,
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    tb = TensorBoard(log_dir=c.LOG_PATH,
                     histogram_freq=10,
                     batch_size=batch_size,
                     write_graph=True,
                     write_grads=True,
                     write_images=False,
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               mode='auto')

    model.fit(network_input, network_output,
              validation_split=0.3,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[tb, early_stop, checkpoint])


if __name__ == '__main__':
    train_network()
