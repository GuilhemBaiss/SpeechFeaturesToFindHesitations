#@Author : Guilhem Baissus
#Algorithm written during an internship at Laboratoire d'ingénierie Cognitive Sémantique (Lincs) located in Montreal, Quebec
#My internship was supervised by Sylvie Ratté

import sys
import os
import io
import pandas as pd
from functions.functions import *

class ExtractFeatures:
    @staticmethod
    def extract_features(pathSound, minimum_silence_duration = 0.1, size_frame = 1.2, size_between_frames = 0.01):
        """
        Method that frames voiced zones of an audio and returns the data extracted. 
        This function calls the method __extract_features_from_frame to get the features from every frame. 
        :params pathSound: path to access to the sound file
        :params minimum_silence_duration: value of the minimum silence duration detection
        :params size_frame: The amount of time per frames
        :params size_between_frames: The amount of time seperating every frame
        :returns: the data extracted from the frames
        """
        # 8 f0 features
        data = {
        "classification" : [],
        "start_time" : [],
        "end_time" : [],
        "duration": [],
        "f0_mean" : [],
        "f0_std" : [],
        "f0_max" : [], 
        "f0_min" : [],
        "f0_reg_coef" : [], 
        "f0_reg_coef_mse" : [],
        "f0_slope" : [],
        "mean_distances_f0" : []
        }

        endSoundFile = SoundFileManipulations.getSoundFileLength(pathSound)
        voicedZones = Silence.detect_voices(pathSound,minimum_silence_duration, 0, endSoundFile)
        silences = Silence.detect_silences(pathSound, minimum_silence_duration, 0, endSoundFile)

        f0_mean_audio = Features_f0.get_f0_mean(pathSound, 0, endSoundFile)

        for _, values in enumerate(voicedZones):
            start_time_frame = values['start_time']
            if values['duration'] < size_frame:
                end_time_frame = values['end_time']
            else:
                end_time_frame = start_time_frame + size_frame

            while end_time_frame <= values['end_time']:
                print("Framing ...")
                if ExtractFeatures.__conditions(pathSound, start_time_frame, end_time_frame, endSoundFile, f0_mean_audio):
                    ExtractFeatures.__extract_features_from_frame( data, pathSound, silences, endSoundFile, start_time_frame, end_time_frame)
                start_time_frame += size_between_frames
                end_time_frame += size_between_frames
            
            last_frame_start_time = end_time_frame - size_between_frames
            if last_frame_start_time != values['end_time'] and values['end_time'] - last_frame_start_time > minimum_silence_duration:
                print("Framing ...")
                ExtractFeatures.__extract_features_from_frame( data, pathSound, silences, endSoundFile, last_frame_start_time, values['end_time'])

        return data

    @staticmethod
    def __extract_features_from_frame( data, pathSound, silences, endSoundFile, start_time_frame, end_time_frame):
        """
        Method that extracts the different features from a specific given frame
        :params data: the dictionnary containing the different lists of all the features
        :params pathSound: path to access to the sound file
        :params silences: array containing all the silences in the given audio
        :params endSoundFile : end time of the entire audio file in seconds
        :params start_frame : starting time of the frame in seconds
        :params end_frame : ending time of the frame in seconds
        """
        print("Extracting features ..." )
        data['classification'].append("Unknown")
        ExtractFeatures.__add_times_features(data, start_time_frame, end_time_frame)
        ExtractFeatures.__add_f0_features(data, pathSound, start_time_frame, end_time_frame)
    
    @staticmethod
    def __conditions(pathSound, start_time_frame, end_time_frame, endSoundFile, f0_mean_audio):
        """
        Conditions to frame
        """
        print("Checking")
        print(start_time_frame, end_time_frame)
        f0_mean_frame = Features_f0.get_f0_mean(pathSound, start_time_frame, end_time_frame)

        reg_coeff_f0 = Features_f0.get_f0_reg_coeff(pathSound, start_time_frame, end_time_frame)
        mean_slope_f0 = Features_f0.get_f0_slope(pathSound, start_time_frame, end_time_frame)

        if (f0_mean_frame < f0_mean_audio or (reg_coeff_f0 < 0 and mean_slope_f0 < 0) or (reg_coeff_f0 > 0 and mean_slope_f0 > 0)) and f0_mean_frame != 0 and Features_f0.check_number_0(pathSound, start_time_frame, end_time_frame):  
            return True      
        return False

    @staticmethod
    def __add_times_features(data, start_time_frame, end_time_frame):
        """
        Method that adds the values linked to time to the dataset
        """
        print("adding times")
        data["start_time"].append(start_time_frame)
        data["end_time"].append(end_time_frame)
        data["duration"].append(end_time_frame - start_time_frame)

    @staticmethod
    def __add_f0_features(data, pathSound, start_time_frame, end_time_frame):
        """
        Method that adds the values linked to f0 to the dataset
        """
        print("adding f0")
        mean = Features_f0.get_f0_mean(pathSound, start_time_frame, end_time_frame)
        data["f0_mean"].append(mean)
        data["f0_std"].append(Features_f0.get_f0_standard_deviation(pathSound, mean, start_time_frame, end_time_frame))
        data["f0_max"].append(Features_f0.get_f0_max(pathSound, start_time_frame, end_time_frame))
        data["f0_min"].append(Features_f0.get_f0_min(pathSound, start_time_frame, end_time_frame))
        data["f0_reg_coef"].append(Features_f0.get_f0_reg_coeff(pathSound, start_time_frame, end_time_frame))
        data["f0_reg_coef_mse"].append(Features_f0.get_f0_squared_error_reg_coeff(pathSound, start_time_frame, end_time_frame))
        data["f0_slope"].append(Features_f0.get_f0_slope(pathSound, start_time_frame, end_time_frame))
        data["mean_distances_f0"].append(Features_f0.get_mean_distance(pathSound, start_time_frame, end_time_frame))
    

#-----------------------------------------------------------------------------------------------------------
#HYPERPARAMETERS 
MINIMUM_SILENCE_DURATION = 0.1
SIZE_FRAME = 0.5
SIZE_BETWEEN_FRAMES = 0.1
#HIGH_OUTLIERS_PERCENTAGE = 5/6
#LOW_OUTLIERS_PERCENTAGE = 0

PATH_SOUND_FILES ="audios"
PATH_CSV_FILES = "csv"

print("python pythonFile.py [path_to_sound_files] [path_to_csv_files]")
#Look if the path is given by argument
if len(sys.argv) ==2:
    PATH_SOUND_FILES = sys.argv[1]

if len(sys.argv) == 3:
    PATH_SOUND_FILES =sys.argv[1]
    PATH_CSV_FILES = sys.argv[2]

elif len(sys.argv) > 3:
    print("Error too many arguments given")
    sys.exit()

data = {}
df = pd.DataFrame()

audio_files_list = os.listdir(PATH_SOUND_FILES)
csv_files_list = os.listdir(PATH_CSV_FILES)

number_audios_processed = 0 
size_audios_folder = len(audio_files_list)

for audios_names in audio_files_list:
    if audios_names[-3:] == "mp3" or audios_names[-3:] == "wav" or audios_names[-3:] == "MP3" or audios_names[-3:] == "WAV" :
        if Functions.check_if_csv_file(csv_files_list, audios_names):
            path_sound_file = Functions.check_os(PATH_SOUND_FILES, audios_names)
            number_audios_processed +=1
            print("Processing file : {} - {}/{}".format(audios_names, number_audios_processed, size_audios_folder))
            data = ExtractFeatures.extract_features(path_sound_file, minimum_silence_duration=MINIMUM_SILENCE_DURATION, size_frame=SIZE_FRAME, size_between_frames=SIZE_BETWEEN_FRAMES)
            df = pd.DataFrame(data,columns=list(data.keys()))
            df.to_csv(PATH_CSV_FILES + "\{}.csv".format(audios_names), index = False)

if number_audios_processed ==0:
    print("The folder given in argument does not contain any sound file or all audios have csv files associated")






