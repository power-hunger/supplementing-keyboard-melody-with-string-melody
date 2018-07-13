""" This files consists of constants used in others modules"""
# Prepare data-set constants
SONG_LIST = '/data/preapared_dataset_data/all_songs.txt'
MIDI_EXCEPTION_FILES = "/data/preapared_dataset_data/exceptions.txt"
MIDI_FILE_DATA = "/data/preapared_dataset_data/midi_file_data.csv"
SONG_DIR_PATH = "/midi_songs"
# Saved all training note
SAVED_KEYB_NOTES = 'data/notes/keyboard_notes'
SAVED_STR_NOTES = 'data/notes/string_notes'
# Lists of instruments
KEYBOARD_INSTRUMENTS = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
STRING_INSTRUMENTS = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                      "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                      "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                      "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto", ]
# Learning model data
LOG_PATH = 'data/logs'
WEIGHTS_PATH = "data/weights/weights.hdf5"
# MIDI files
MIDI_OUTPUT_PATH_K = "data/generated_midi/output_k.mid"
MIDI_OUTPUT_PATH_S = "data/generated_midi/output_s.mid"
MIDI_SEED_PATH = "data/seed.mid"
