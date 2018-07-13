""" This module sorts data for training from The Lakh MIDI Dataset v0.1"""
from music21 import chord, converter, instrument, note, stream, common
from pathlib import Path
import constants as c
import csv


def prepare_data():
    path_list = []
    for midi_file_path in Path(c.SONG_DIR_PATH).glob('**/*.mid'):
        path_list.append(midi_file_path)
    common.runParallel(path_list, check_instruments_and_save_notes)


def check_instruments_and_save_notes(midi_file_path):
    try:
        # Some MIDI files will raise Exceptions on loading, if they are invalid we just skip those.
        song = converter.parse(midi_file_path)
        # Extract MIDI file with two instruments where one is piano
        parts = instrument.partitionByInstrument(song)
        if len(parts) == 2:
            if (parts.parts[0].id in c.KEYBOARD_INSTRUMENTS and parts.parts[1].id in c.STRING_INSTRUMENTS) or \
                    (parts.parts[1].id in c.KEYBOARD_INSTRUMENTS and parts.parts[0].id in c.STRING_INSTRUMENTS):

                percent_diff, k_note_count, s_note_count = get_midi_file_info(midi_file_path)
                write_to_csv(c.MIDI_FILE_DATA,
                             percent_diff,
                             k_note_count,
                             s_note_count,
                             midi_file_path)
                # In our case we will train our network on notes where note diff is not more than 10%
                if percent_diff <= 0.1:
                    write_to_file(c.SONG_LIST, midi_file_path)

    except Exception as e:
        print("Exception ", midi_file_path, e)
        write_to_file(c.MIDI_EXCEPTION_FILES, midi_file_path + " " + e)
        pass


def get_midi_file_info(midi_file_path):
    """ Get notes and calculate their difference in percents """
    keyboard_notes = get_notes_chords_rests(c.KEYBOARD_INSTRUMENTS, midi_file_path)
    string_notes = get_notes_chords_rests(c.STRING_INSTRUMENTS, midi_file_path)

    notes_max = max(len(keyboard_notes), len(string_notes))
    notes_min = min(len(keyboard_notes), len(string_notes))
    percent_diff = (notes_max - notes_min) / notes_max

    print(midi_file_path +
          "\n k_note_amount: " + str(len(keyboard_notes)) +
          "\n s_note_amount: " + str(len(string_notes)))

    return percent_diff, len(keyboard_notes), len(string_notes)


def get_notes_chords_rests(instrument_type, path):
    """ Get all the notes, chords and rests from the midi files in the ./midi_songs directory """
    note_list = []
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
    """Check if there is not missing rests which ensures silent part in melody"""
    if element.offset / 0.5 <= len(note_list):
        return
    else:
        while True:
            note_list.append('Rest')
            if element.offset / 0.5 <= len(note_list):
                return


def write_to_file(file_path, what_to_write):
    text_file = open(file_path, "a+")
    text_file.write(str(what_to_write) + "\n")
    text_file.close()


def write_to_csv(csv_file_path, percentage, note_p, note_s, midi_file_path):
    with open(str(csv_file_path), 'a+', newline='') as csvfile:
        fieldnames = ['Percentage', 'Note_P', 'Note_S', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Percentage': str(percentage),
                         'Note_P': str(note_p),
                         'Note_S': str(note_s),
                         'Path': str(midi_file_path)})


def init_csv_file(csv_file_path):
    with open(str(csv_file_path), 'a+', newline='') as csvfile:
        fieldnames = ['Percentage', 'Note_P', 'Note_S', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


if __name__ == '__main__':
    init_csv_file(c.MIDI_FILE_DATA)
    prepare_data()
