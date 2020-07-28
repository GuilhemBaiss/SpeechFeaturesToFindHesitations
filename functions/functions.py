
import srt
import parselmouth
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class SoundFileManipulations:

    @staticmethod
    def getSoundFileLength(pathSound):
        """
        Function that returns the length of a sound file in seconds
        :params pathMP3 : the path to the sound file
        :returns: the length of sound file in seconds
        """
        sound = parselmouth.Sound(pathSound)
        return sound.xmax - sound.xmin

class Silence:

    @staticmethod
    def detect_voices(pathSound, minimum_silence_duration, start, end, frame = 20):
        """
        Method that detects voiced zones for long and short audio using the method that detects silences
        :params pathSound: path to access to the audio 
        :params minimum_silence_duration: the length of the silence pauses to detect and the minimum length of voiced zones
        :params start: parameter of where to start checking for voices in a given audio
        :params end : parameter of where to end checking for voices in a given audio
        :params frame : size of a frame that we check for silences
        :returns: an array containing dictionnaries describing the start time, the end time and the duration of a the voiced zones
        """
        voices = []
        silences = Silence.detect_silences(pathSound, minimum_silence_duration, start, end, frame)
        for index, values in enumerate(silences):
            #if there is voices at the beginning of the audio before the first silence
            if index ==0 and values['start_time'] > 0.5:
                duration = values['start_time'] - start
                if duration >= minimum_silence_duration:
                    voices.append({'start_time' : start, 'end_time' : values['start_time'], 'duration' : duration})   

            #if there is voices during the audio
            if index + 1 < len(silences) - 1:
                duration = silences[index + 1]['start_time'] - values['end_time']
                if duration >= minimum_silence_duration:
                    voices.append({'start_time' : values['end_time'], 'end_time' : silences[index+1]['start_time'], 'duration' : duration})

            #if the audio does not finish by a silence
            if index == len(silences) - 1:
                duration = end - values['end_time']
                if duration >= minimum_silence_duration:
                    voices.append({'start_time' : values['end_time'], 'end_time' : end, 'duration' : duration})

        return voices

    @staticmethod
    def detect_silences(pathSound, minimum_silence_duration, start, end, frame = 20):
        """
        Method that detects silences for long and short audio 
        :params pathSound: path to access to the audio 
        :params minimum_silence_duration: the length of the silence pauses to detect
        :params start: parameter of where to start checking for silences in a given audio
        :params end : parameter of where to end checking for silences in a given audio
        :params frame : size of a frame that we check for silences
        :returns: an array containing dictionnaries describing the start time, the end time and the duration of a silent pause
        """
        length_extracted_audio = end - start
        outliers = Features_f0.get_outliers(pathSound)
        low_outliers_value = outliers[0]
        high_outliers_value =  outliers[1]
        silences = []

        if(length_extracted_audio>frame):
            nb_of_frames = int(length_extracted_audio/frame)
            start_frame = start
            end_frame = start_frame + frame
            for _ in range(1, nb_of_frames+1):
                silences += Silence.__detect_silence_in_frame(pathSound, start_frame, end_frame, minimum_silence_duration, high_outliers_value, low_outliers_value)
                start_frame +=frame
                end_frame +=frame

            last_end_frame = end_frame - frame
            if last_end_frame < end and end - last_end_frame > 1:
                #Last frame that is not equal to the frame length
                silences+= Silence.__detect_silence_in_frame(pathSound, last_end_frame, end, minimum_silence_duration, high_outliers_value, low_outliers_value)
        else:
            silences += Silence.__detect_silence_in_frame(pathSound, start, end, minimum_silence_duration, high_outliers_value, low_outliers_value)

        return silences

    @staticmethod
    def __detect_silence_in_frame(pathSound, start_frame, end_frame, minimum_silence_duration, high_outliers_value, low_outliers_value):
        """
        Method that detects silences in a given frame using the fundamental frequency and removes outliers using mean and standard deviation
        :params pathSound: path to access to the audio 
        :params start_frame: parameter of where to start checking for silences in a given audio
        :params end_frame : parameter of where to end checking for silences in a given audio
        :params minimum_silence_duration: the length of the silence pauses to detect
        :returns: an array containing dictionnaries describing the start time, the end time and the duration of a silent pause
        """
        silences = []

        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_frame , to_time = end_frame + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        start_time_silence = -1
        end_time_silence = -1
        duration = -1
        pauseState = False

        for index, values in enumerate(pitch_values):
            if values >= high_outliers_value or values <= low_outliers_value:
                if pauseState == False:
                    start_time_silence = pitch.xs()[index]
                    pauseState = True
                #Check if there is silence at the end of the audio
                elif pauseState == True and pitch.xs()[index] == pitch.xs()[-1] and (values > high_outliers_value or values < low_outliers_value):
                    silences.append({'start_time': start_frame + start_time_silence, 'end_time': start_frame + pitch.xs()[index], 'duration': pitch.xs()[index] - start_time_silence})
            else:
                if pauseState == True :
                    end_time_silence = pitch.xs()[index]
                    duration = end_time_silence - start_time_silence
                    if duration > minimum_silence_duration:
                        silences.append({'start_time': start_frame + start_time_silence, 'end_time': start_frame + end_time_silence, 'duration': duration})        
            
                pauseState = False

        return silences
            
class Features_f0:

    @staticmethod
    def get_f0_mean(pathSound, start_time, end_time, voice_max_frequency = 300, voice_min_frequency=50):
        """
        Method that extracts the f0 mean of a particular sound found at pathMP3 location without taking in count the 0 values. 
        :params pathSound: path to the sound to analyse
        params voice_max_frequency : maximum frequency of a human being (adult man or adult female)
        params voice_min_frequency : minimum frequency of a human being (adult man or adult female)
        """
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_mean = 0
        size= 1
        for values in pitch_values:
            if values > voice_min_frequency and values < voice_max_frequency:
                pitch_mean += values
                size +=1

        return pitch_mean/size

    @staticmethod   
    def get_f0_q1_median_q3(pathSound, start_time, end_time, low_outliers_percentage = 0, high_outliers_percentage = 5/6):
        """
        Get the median, the first quartil and the third quartil to establish a box plot
        :params pathSound: path to the sound to analyse
        :returns: median, first quartil and third quartil
        """
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values.sort()
        array =[]
        size = 0
        for values in pitch_values:
            if values!=0:
                size +=1
                array.append(values)
        
        index_q1 = low_outliers_percentage
        index_median = size / 2
        index_q3 = size * high_outliers_percentage
        return array[int(index_q1)], array[int(index_median)], array[int(index_q3)]
    

    @staticmethod
    def get_f0_standard_deviation(pathSound, mean, start_time, end_time, voice_max_frequency = 300, voice_min_frequency=50):
        """
        Get the standard deviation around a mean
        :params pathSound: path to the sound to analyse
        params voice_max_frequency : maximum frequency of a human being (adult man or adult female)
        params voice_min_frequency : minimum frequency of a human being (adult man or adult female)
        :returns: standard deviation of the sound
        """
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        sum = 0
        size = 0
        for values in pitch_values:
            if values > voice_min_frequency and values < voice_max_frequency:
                sum += math.pow(values - mean,2)
                size += 1
        
        return math.sqrt(sum / size)
    
    @staticmethod
    def get_f0_reg_coeff(pathSound, start_time, end_time):
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_x = pitch.xs()
        pitch_y = pitch.selected_array['frequency']
        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        low_outliers = outliers[0]
        high_outliers = outliers[1]
        X = []
        Y = []
        #Remove values that are outliers
        for index, values in enumerate(pitch_y):
            if values >= low_outliers and values <= high_outliers:
                Y.append(values)
                X.append(pitch_x[index])

        X_data = np.array(X).reshape(-1,1)
        Y_data = np.array(Y).reshape(-1,1)
        
        model = LinearRegression()
        model.fit(X_data, Y_data)
        y_pred = model.predict(X_data)

        # plt.scatter(X_data, Y_data,  color='gray')
        # plt.plot(X_data, y_pred, color='red', linewidth=2)
        # plt.show()

        return model.coef_[0][0]

    @staticmethod
    def get_f0_squared_error_reg_coeff(pathSound, start_time, end_time):
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_x = pitch.xs()
        pitch_y = pitch.selected_array['frequency']
        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        low_outliers = outliers[0]
        high_outliers = outliers[1]
        X = []
        Y = []
        #Remove values that are outliers
        for index, values in enumerate(pitch_y):
            if values >= low_outliers and values <= high_outliers:
                Y.append(values)
                X.append(pitch_x[index])

        X_data = np.array(X).reshape(-1,1)
        Y_data = np.array(Y).reshape(-1,1)

        model = LinearRegression()
        model.fit(X_data, Y_data)

        Y_pred = model.predict(X_data)

        return metrics.mean_squared_error(Y_data, Y_pred)


    @staticmethod 
    def get_f0_max(pathSound, start_time, end_time):
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        high_outliers =  outliers[1]

        maximum = np.amax(pitch_values)
        if maximum < high_outliers :
            return maximum
        else:
            maximum = -100
            for _, values in enumerate(pitch_values):
                if values > maximum and values < high_outliers and values != 0:
                    maximum = values
            return maximum
    
    @staticmethod
    def get_f0_min(pathSound, start_time, end_time):
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        low_outliers =  outliers[0]

        minimum = np.amin(pitch_values)
        if minimum > low_outliers:
            return minimum
        else:
            minimum = 100000
            for _, values in enumerate(pitch_values):
                if values < minimum and values> low_outliers:
                    minimum = values
            return minimum

    @staticmethod
    def get_f0_slope(pathSound, start_time, end_time):
        """
        Method that returns the slope of f0 on a given window of an audio file
        :params pathSound: path to the sound to analyse
        """
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        low_outliers = outliers[0]
        high_outliers = outliers[1]

        sum = 0
        size = 0
        for index in range(0, len(pitch_values)):
            if index+1 < len(pitch_values) and pitch_values[index] >= low_outliers and pitch_values[index + 1] >= low_outliers and pitch_values[index]<= high_outliers and pitch_values[index + 1] <= high_outliers:
                #print ("i : {} i+1 : {}".format(pitch_values[index], pitch_values[index + 1]))
                #print("s : {} s+1: {}".format(pitch.xs()[index], pitch.xs()[index +1]))
                sum += (pitch_values[index + 1] - pitch_values[index] ) / (pitch.xs()[index +1] - pitch.xs()[index])
                #print("sum :  {}".format((pitch_values[index + 1] - pitch_values[index] ) / (pitch.xs()[index +1] - pitch.xs()[index])))
                size += 1

        if size != 0:
            slope = sum / size
            return slope
        else:
            return 0
    
    @staticmethod
    def get_outliers(pathSound, voice_max_frequency = 300, voice_min_frequency=50):
        """
        Method that returns the borders to where to analyse the voice signal
        params pathSound : path to the audio file
        params voice_max_frequency : maximum frequency of a human being (adult man or adult female)
        params voice_min_frequency : minimum frequency of a human being (adult man or adult female)
        """
        outliers = []
        length_audio = SoundFileManipulations.getSoundFileLength(pathSound)

        mean_entire_audio = Features_f0.get_f0_mean(pathSound, 0, length_audio, voice_max_frequency, voice_min_frequency)
        standard_deviation_entire_audio = Features_f0.get_f0_standard_deviation(pathSound, mean_entire_audio, 0, length_audio, voice_max_frequency, voice_min_frequency)

        high_outliers_value = mean_entire_audio + 4*standard_deviation_entire_audio
        if high_outliers_value > voice_max_frequency:
            high_outliers_value = voice_max_frequency

        low_outliers_value = mean_entire_audio - 2*standard_deviation_entire_audio
        if low_outliers_value < voice_min_frequency:
            low_outliers_value = voice_min_frequency

        outliers.append(low_outliers_value)
        outliers.append(high_outliers_value)

        return outliers

    @staticmethod
    def get_outliers_in_frame(pathSound, start_time, end_time, voice_max_frequency = 300, voice_min_frequency=50):
        """
        Method that returns the borders to where to analyse the voice signal
        params pathSound : path to the audio file
        params voice_max_frequency : maximum frequency of a human being (adult man or adult female)
        params voice_min_frequency : minimum frequency of a human being (adult man or adult female)
        """
        outliers = []
        q1, median, q2 = Features_f0.get_f0_q1_median_q3(pathSound, start_time, end_time)
        
        high_outliers_value = q2
        if high_outliers_value > voice_max_frequency:
            high_outliers_value = voice_max_frequency

        low_outliers_value = q1
        if low_outliers_value < voice_min_frequency:
            low_outliers_value = voice_min_frequency

        outliers.append(low_outliers_value)
        outliers.append(high_outliers_value)

        return outliers
    @staticmethod
    def get_mean_distance(pathSound, start_time, end_time):
        """
        Method that returns the distances between the f0 points during a given time
        params pathSound : path to the audio file
        params start_time : where to start analysing
        params end_time : where to end analysing
        """
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        outliers = Features_f0.get_outliers_in_frame(pathSound, start_time, end_time)
        low_outliers =  outliers[0]
        high_outliers = outliers[1]
        frequencies = []
        times = []
        mean_distance = 0

        for index, values in enumerate(pitch_values):
            if values >= low_outliers and values <= high_outliers:
                frequencies.append(values)
                times.append(pitch.xs()[index])
                if len(frequencies) >= 2:
                    mean_distance += math.sqrt(math.pow(values - frequencies[len(frequencies) - 2], 2 ) + math.pow(times[len(frequencies) -1] - times[len(frequencies)-2],2))
                    
        return mean_distance / len(frequencies)

    @staticmethod
    def check_number_0(pathSound, start_time, end_time):
        sound = parselmouth.Sound(pathSound)
        sound = sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        if np.count_nonzero(pitch_values) > 2:
            return True
        else:
            return False

    @staticmethod
    def plot_f0(pathSound, start_time, end_time):
        """
        Plot the fundamental frequencies of a sound
        :params pathSound: path to the sound to analyse
        """
        sound = parselmouth.Sound(pathSound)
        sound= sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        plt.figure()
        pitch_values[pitch_values==0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        plt.ylabel("fundamental frequency [Hz]")
        plt.xlim([sound.xmin, sound.xmax])
        plt.show()

class Features_energy:
    @staticmethod
    def get_array_energy(pathSound, start_time, end_time, number_samples = 100):
        sound = parselmouth.Sound(pathSound)
        sound= sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        amplitudes = sound.values.T
        times = sound.xs()

        size_between_frames = int(len(amplitudes) /number_samples)
        size_frame = int(size_between_frames * 4)

        energy = []
        energy_sample = 0
        end_time = times[-1]
        start_frame = 0
        end_frame = size_frame

        while end_frame <= len(amplitudes):
            for _, values in enumerate(amplitudes[start_frame:end_frame]):
                energy_sample += math.pow(values[0], 2)
            energy.append([(times[start_frame] + times[end_frame - 1])/2 , energy_sample])
            energy_sample = 0
            start_frame += size_between_frames
            end_frame += size_between_frames

        #if the last bit is smaller than the size_between_frames
        last_end_frame = end_frame - size_between_frames
        last_start_frame = start_frame - size_between_frames
        if(last_end_frame != len(amplitudes)):
            for _, values in enumerate(amplitudes[last_start_frame :]):
                energy_sample += math.pow(values[0], 2)
            energy.append([ (times[last_start_frame] + times[last_end_frame]) /2 , energy_sample])
            
        # plt.figure()
        # plt.plot(sound.xs(), sound.values.T)
        # plt.xlim([sound.xmin, sound.xmax])
        # plt.xlabel("time [s]")
        # plt.ylabel("amplitude")
        # plt.show()
        return energy

    @staticmethod
    def get_max_energy(array_energy):
        return np.amax(array_energy)

    @staticmethod
    def get_min_energy(array_energy):
        return np.amin(array_energy)

    @staticmethod
    def get_mean_energy(array_energy):
        return np.mean(array_energy)
    @staticmethod
    def get_energy_reg_coeff(array_energy):
        """
        Explicit
        """
        X = []
        Y = []

        for values in array_energy:
            X.append(values[0])
            Y.append(values[1])

        X_data = np.array(X).reshape(-1,1)
        Y_data = np.array(Y).reshape(-1,1)

        model = LinearRegression()
        model.fit(X_data, Y_data)
        y_pred = model.predict(X_data)

        # plt.scatter(X_data, Y_data,  color='gray')
        # plt.plot(X_data, y_pred, color='red', linewidth=2)
        # plt.show()

        return model.coef_[0][0]

    @staticmethod
    def get_energy_squared_error_reg_coeff(array_energy):
        """
        Explicit
        """
        X = []
        Y = []
        for values in array_energy:
            X.append(values[0])
            Y.append(values[1])

        X_data = np.array(X).reshape(-1,1)
        Y_data = np.array(Y).reshape(-1,1)

        model = LinearRegression()
        model.fit(X_data, Y_data)

        Y_pred = model.predict(X_data)

        return metrics.mean_squared_error(Y_data, Y_pred)

class Features_phonetic:
    # @staticmethod
    # def draw_spectogram(pathSound, start_time, end_time, dynamic_range = 70):
    #     """
    #     Methods that draws a spectogram
    #     params pathSound : path to the audio file to analyse
    #     params start_time : where to start analysing
    #     params end_time : where to end analysing
    #     params dynamic_range : don't really know what this is
    #     """
    #     sound = parselmouth.Sound(pathSound)
    #     sound= sound.extract_part(from_time = start_time , to_time = end_time)
    #     spectrogram = sound.to_spectrogram()
    #     plt.figure()
    #     X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    #     sg_db = 10 * np.log10(spectrogram.values)
    #     plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    #     plt.ylim([spectrogram.ymin, spectrogram.ymax])
    #     plt.xlabel("time [s]")
    #     plt.ylabel("frequency [Hz]")
    #     plt.xlim([sound.xmin, sound.xmax])
    #     plt.show()

    @staticmethod
    def get_mfcc_coefficients(pathSound, start_time, end_time):
        """
        Method that extracts number of coefficients + 1 coefficients in samplings
        params pathSound : path to the audio file to analyse
        params start_time : where to start analysing the audio
        params end_time : where to end analysing the audio
        returns : an array of size number of coefficients + 1 of arrays representing every value of each sampling for a particular coefficient 
        """
        number_of_coefficients = 12

        sound = parselmouth.Sound(pathSound)
        sound= sound.extract_part(from_time = start_time , to_time = end_time + 0.02)
        coefficients = sound.to_mfcc(number_of_coefficients)
        coefficients = coefficients.to_array()
        return coefficients

    @staticmethod
    def get_means_coeffs(coefficients):
        """
        Method that returns the mean of every coefficient given in the coefficient parameter
        params coefficients: array returned in get_mfcc_coefficients()
        returns : an array of size of the number of coefficients in the array coefficient given as a parameter
        """
        number_of_coefficients = len(coefficients)
        size_sampling = len(coefficients[0])
        means = []
        
        for i in range(0, number_of_coefficients):
            mean_coef = 0
            for values in coefficients[i]:
                mean_coef += values
            means.append(mean_coef /size_sampling)
        return means

    @staticmethod
    def get_standard_deviation_coeffs(coefficients):
        """
        Method that returns the standart deviation of every coefficient given in the coefficient parameter
        params coefficients: array returned in get_mfcc_coefficients()
        returns : an array of size of the number of coefficients in the array coefficient given as a parameter
        """
        number_of_coefficients = len(coefficients)
        size_sampling = len(coefficients[0])
        mean = Features_phonetic.get_means_coeffs(coefficients)
        standard_deviations = []

        for i in range(0, number_of_coefficients):
            variance = 0
            for values in coefficients[i]:
                variance += math.pow((values - mean[i])/size_sampling,2)
            standard_deviations.append(math.sqrt(variance))
        
        return standard_deviations

    @staticmethod
    def get_minimum_stability_distance(coefficients, number_of_distances):

        number_of_values_per_coefficients =  len(coefficients[0])
        if number_of_distances > number_of_values_per_coefficients:
            number_of_distances = number_of_values_per_coefficients
        number_of_values_per_distance = int( number_of_values_per_coefficients/ number_of_distances) 

        index_start_frame = number_of_values_per_distance
        index_end_frame = 2 * number_of_values_per_distance
        minimum_stability_distance = 10000
        while index_start_frame < number_of_values_per_coefficients - 2*number_of_values_per_distance:

            distance = Features_phonetic.get_stability_distance(coefficients, index_start_frame, index_end_frame, number_of_values_per_distance)
            if distance < minimum_stability_distance:
                minimum_stability_distance = distance

            index_start_frame += number_of_values_per_distance
            index_end_frame += number_of_values_per_distance

        return minimum_stability_distance

    @staticmethod
    def get_stability_distance(coefficients, index_start_frame, index_end_frame, number_values_per_distance):
        """
        Method that calculates the stability distance of one frame. It s the sum of the distances of each coefficients of the frame before and the frame after compared to the actual frame divided by 2
        """
        #get the specific coefficients per frame
        coefficients_previous_frame = []
        coefficients_frame = []
        coefficients_next_frame = []
        
        for index in range(0,len(coefficients)):
            coefficients_previous_frame.append(coefficients[index][index_start_frame - number_values_per_distance : index_start_frame])
            coefficients_frame.append(coefficients[index][index_start_frame : index_end_frame])
            coefficients_next_frame.append(coefficients[index][index_end_frame :  index_end_frame + number_values_per_distance])

        #get mean coefficients per frame
        mean_coefficients_previous_frame = Features_phonetic.get_means_coeffs(coefficients_previous_frame)
        mean_coefficients_frame = Features_phonetic.get_means_coeffs(coefficients_frame)
        mean_coefficients_next_frame = Features_phonetic.get_means_coeffs(coefficients_next_frame)
        
        #get distances between frames
        distance_previous_frame = 0
        distance_next_frame = 0

        for index,values in enumerate(mean_coefficients_frame):
            distance_previous_frame += math.pow(values - mean_coefficients_previous_frame[index],2)
            distance_next_frame += math.pow(values - mean_coefficients_next_frame[index],2)
        
        distance_previous_frame = math.sqrt(distance_previous_frame)
        distance_next_frame = math.sqrt(distance_next_frame)

        return (distance_previous_frame + distance_next_frame)/2

class Functions:

    @staticmethod
    def check_os(path, audios_names):
        """
        Method that returns the path to a precise audio file depending of the user's operating system
        params path : path to the folder containing the sound files
        params audios_name : name of the sound file that the algorithm is going to analyse
        returns : path to the specific audio file. 
        """
        if "/" in path and "\\" is not True:
            path_sound_file = path + "/" + audios_names
        else:
            path_sound_file = path + "\\" + audios_names
        return path_sound_file

    


