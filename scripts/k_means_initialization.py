import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2


class KMeansInitialization:
    def __init__(self, image, bounding_boxes):
        self.image = image
        self.bounding_boxes = bounding_boxes
        self.data_table = None

    def process_data(self):
        dic_of_data = {"r": [], "g": [], "b": [], "class": []}
        for bb_coordinates, bb_class in self.bounding_boxes:
            top_left_x, top_left_y, width, height = bb_coordinates
            image_slice = self.image[top_left_y: top_left_y+height, top_left_x: top_left_x+width]
            
            for i in range(image_slice.shape[0]):
                for j in range(image_slice.shape[1]):
                    dic_of_data["r"].append(image_slice[i][j][0])
                    dic_of_data["g"].append(image_slice[i][j][1])
                    dic_of_data["b"].append(image_slice[i][j][2])
                    dic_of_data["class"].append(bb_class)
        
        self.data_table = pd.DataFrame(dic_of_data)

    def run_kmeans_sklearn(self, n_clusters=2):
        X = self.data_table[['r', 'g', 'b']].values  

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data_table["assigned_component"] = kmeans.fit_predict(X)

        print(f"K-Means finished with {n_clusters} clusters.")


    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)


    def run_kmeans(self, epsilon=0.001, iterations_upper_threshold=100):
        old_kmean_1_random_mean = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        old_kmean_2_random_mean = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        new_kmean_1_random_mean = (0, 0, 0)
        new_kmean_2_random_mean = (0, 0, 0)
        iterations = 0

        while (self.euclidean_distance(old_kmean_1_random_mean, new_kmean_1_random_mean) > epsilon) or (self.euclidean_distance(old_kmean_2_random_mean, new_kmean_2_random_mean) > epsilon) and iterations <iterations_upper_threshold:
            old_kmean_1_random_mean = new_kmean_1_random_mean
            old_kmean_2_random_mean = new_kmean_2_random_mean
            assigned_components = []

            for i in range(len(self.data_table)):
                distance_to_mean_1 = self.euclidean_distance((self.data_table.iloc[i]["r"], self.data_table.iloc[i]["g"], self.data_table.iloc[i]["b"]), old_kmean_1_random_mean)
                distance_to_mean_2 = self.euclidean_distance((self.data_table.iloc[i]["r"], self.data_table.iloc[i]["g"], self.data_table.iloc[i]["b"]), old_kmean_2_random_mean)
                if distance_to_mean_1 < distance_to_mean_2:
                    assigned_components.append(1)
                else:
                    assigned_components.append(2)

            self.data_table["assigned_component"] = assigned_components

            new_kmean_1_random_mean = (self.data_table[self.data_table["assigned_component"] == 1]["r"].mean(),
                                self.data_table[self.data_table["assigned_component"] == 1]["g"].mean(),
                                self.data_table[self.data_table["assigned_component"] == 1]["b"].mean())
            
            new_kmean_2_random_mean = (self.data_table[self.data_table["assigned_component"] == 2]["r"].mean(),
                                self.data_table[self.data_table["assigned_component"] == 2]["g"].mean(),
                                self.data_table[self.data_table["assigned_component"] == 2]["b"].mean())

            iterations +=1
            print(f"Iteration {iterations}")
        print(f"Finished kmeans in {iterations} iterations")
            

    def run(self):
        print("Starting with processing the data...")
        self.process_data()
        print("Finished the processing of the data.")
        print("Running k-means clustering...")
        self.run_kmeans_sklearn()
        # self.run_kmeans()
        print("Finished k-means clustering.")
        return self.data_table
