""" This module generates notes for a midi file using the trained neural network """
import numpy
import pickle
import constants as c
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from music21 import instrument, note, stream, chord, converter


def generate():
    """ Generate new midi file which is corresponding to given one """
    # Load the notes
    with open(c.SAVED_KEYB_NOTES, 'rb') as file_path:
        keyboard_notes = pickle.load(file_path)
    with open(c.SAVED_STR_NOTES, 'rb') as file_path:
        string_notes = pickle.load(file_path)
    # Get all pitch names
    pitch_names_k = sorted(set(item for item in keyboard_notes))
    pitch_names_s = sorted(set(item for item in string_notes))
    # Get all pitch name count
    n_vocab_k = len(set(keyboard_notes))
    n_vocab_s = len(set(string_notes))

    k_seed_notes = get_notes_chords_rests(c.MIDI_SEED_PATH)

    network_input, one_hot_en_input = prepare_sequences(k_seed_notes, pitch_names_k)
    model = create_network(one_hot_en_input, n_vocab_s)
    k_notes, s_notes = generate_notes(model, network_input, pitch_names_k, pitch_names_s, n_vocab_k)
    create_midi(k_notes, c.MIDI_OUTPUT_PATH_K)
    create_midi(s_notes, c.MIDI_OUTPUT_PATH_S)


def get_notes_chords_rests(path):
    """ Get all the notes, chords and rests from the midi files in the song directory """
    note_list = []
    try:
        midi_file = converter.parse(path)
        for el in midi_file.recurse().notes:
            if isinstance(el, note.Note):
                check_rest_amount(el, note_list)
                note_list.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                check_rest_amount(el, note_list)
                note_list.append('.'.join(str(n) for n in el.normalOrder))
            elif isinstance(el, note.Rest):
                check_rest_amount(el, note_list)
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


def prepare_sequences(notes, pitch_names):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = len(notes)-1
    network_input = []

    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    # create a dictionary to map pitches to integers
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input, one-hot-encoding
    normalized_input = np_utils.to_categorical(normalized_input)

    return network_input, normalized_input


def create_network(network_input, n_vocab_s):
    """ create the structure of the neural network """
    model = Sequential()

    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], 567),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dense(256)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(n_vocab_s, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights(c.WEIGHTS_PATH)

    return model


def generate_notes(model, network_input, pitch_names_k, pitch_names_s, n_vocab_k):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    prediction_output = []
    seed_k_notes = []
    # start = numpy.random.randint(0, len(network_input)-100)
    pattern = network_input[0]
    int_to_note_k = dict((number, note) for number, note in enumerate(pitch_names_k))
    int_to_note_s = dict((number, note) for number, note in enumerate(pitch_names_s))

    prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
    prediction_input = np_utils.to_categorical(prediction_input, n_vocab_k)

    prediction = model.predict(prediction_input, verbose=0)
    prediction = (numpy.argmax(prediction, axis=2)).flatten()

    for note_s in prediction.tolist():
        prediction_output.append(int_to_note_s[note_s])
    for note_k in pattern:
        seed_k_notes.append(int_to_note_k[note_k])

    return seed_k_notes, prediction_output


def create_midi(note_list, path):
    """ convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in note_list:
        # pattern is a Rest
        if pattern == 'Rest':
            new_note = note.Rest()
            new_note.offset = offset
            output_notes.append(new_note)
        # pattern is a chord
        elif ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=path)


if __name__ == '__main__':
    generate()
