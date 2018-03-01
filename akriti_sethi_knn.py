# To run the main program demonstrating the KNN algorithm, run this file on the cmd
# To run the graph part of the program, please comment out the call for read_file() and uncomment the call for createGraph()

import zipfile
import urllib
from operator import itemgetter
import matplotlib.pyplot as plt


class Environment:
    k = 1

    def __init__(self):
        pass

    def read_file(self):
        input_file_name = raw_input("Kindly enter the folder name you wish to implement KNN for\n")

        while (True):
            k = raw_input("Please enter an odd value of k\n")
            if int(k) % 2 is 0:
                continue
            else:
                break

        print(".........................Computation in progress...................")
        # Download the zip file from the url based on foldere name
        urllib.urlretrieve("http://sci2s.ugr.es/keel/dataset/data/classification/" + input_file_name + "-10-fold.zip",
                           input_file_name + "-10-fold.zip")

        # Extract the zip file into a folder
        data_file = zipfile.ZipFile(input_file_name + "-10-fold.zip", 'r')
        data_file.extractall(input_file_name + "-10-fold")

        # sending the value of k to the Agent
        agent_obj = Agent(k, input_file_name)
        accuracy = []

        for i in range(1, 11):
            # sending training data file name
            agent_obj.receive_tra_file_name(input_file_name + "-10-"+str(i)+"tra.dat")

            # Test data set
            arrTest = []
            with open(input_file_name + "-10-fold/" + input_file_name + "-10-"+str(i)+"tst.dat", 'r') as f:
                for line in f:
                    arrTest.append(list(line.split()))

            # Removing the rows upto the records from the dat file
            z = 0
            while True:
                if len(arrTest[z]) is 1:
                    break
                else:
                    z = z+1

            arrTest = arrTest[z+1:]
            # arrTest contains the records from the tst.dat file stored in the form of an array

            correct_predictions = 0

            for j in range(0, len(arrTest)):
                # sending each individual percept to the Agent
                # receiving predicted class for each percept
                predicted_output = agent_obj.sensor(arrTest[j])

                # comparing predicted and actual class
                if (predicted_output[1]) == (arrTest[j][len(arrTest[j])-1]):
                    correct_predictions = correct_predictions + 1

            # calculate accuracy for i th file
            accuracy.append((correct_predictions/float(len(arrTest)))*100)

        print("The accuracy for each train and test file is : ")
        for i in range(0, len(accuracy)):
            print(str(i+1)+"        "+str(round(accuracy[i], 2)))

        data_file.close()


class Agent:
    k = 0
    input_file_name = ''
    tra_file_name = ''

    # initializing the value of k which is received from the environment
    def __init__(self, k_value, input_file):
        self.k = k_value
        self.input_file_name = input_file

    # initializing the training file name
    def receive_tra_file_name(self, file_name):
        self.tra_file_name = file_name

    def sensor(self, arrTest):
        # arrTest contains the j th record of the i th test file and is in the form of an array
        # processing Train data file to put it into an array (lookup table)

        arrTrain = []
        with open(self.input_file_name + "-10-fold/" + self.tra_file_name, 'r') as f:
            for line in f:
                arrTrain.append(list(line.split()))

        # Removing the rows upto the records from the dat file
        z = 0
        while True:
            if len(arrTrain[z]) is 1:
                break
            else:
                z = z + 1

        arrTrain = arrTrain[z + 1:]

        # percept sequence passed to function component
        final_output = self.function_component(arrTrain, arrTest)
        return final_output

    def function_component(self, arrTrain, arrTest):

        distance = []
        corr_train_value = []

        for i in range(0, len(arrTrain)):
            # storing the distance between the train and test points
            distance.append(self.compute_distance(arrTest, arrTrain[i]))
            # Storing the corresponding train output class for the points. Only the output field is picked since
            # the train dataset has no value to us
            corr_train_value.append(arrTrain[i][len(arrTrain[i])-1])

        # Combining the two lists into a 2D list
        main_list = zip(distance, corr_train_value)

        # sorting the combined list on the basis of distance in ascending order
        main_list = sorted(main_list, key=itemgetter(0))

        # extracting the k nearest neighbours
        main_list = main_list[0: int(self.k)]

        # passing the list of k nearest neighbours to the actuator method
        return self.actuator(main_list)

    # This method will classify each percept to the best output field attribute
    def actuator(self, main_list):
        max_count = 0

        # To find the frequency of each output attribute
        for i in range(0, len(main_list)):
            count = 1
            for j in range(i + 1, len(main_list)):
                if main_list[i] is main_list[j]:
                    count = count + 1
            if count > max_count:
                corr_value = main_list[i]
                max_count = max(max_count, count)
            i = i + count

        return corr_value

    # Compute the distance between points
    def compute_distance(self, dataset1, dataset2):
        distance1 = 0
        i = 0
        while i < (len(dataset2)-1):
            try:
                data1 = float(dataset1[i].replace(',', ''))
            except:
                data1 = float(dataset1[i])
            data2 = float(dataset2[i].replace(',', ''))
            distance1 = distance1 + pow((data2 - data1), 2)
            i = i+1
        distance1 = round(distance1, 2)
        return distance1


env_obj = Environment()
env_obj.read_file()
#env_obj.createGraph()
