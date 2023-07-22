# Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# Define a function for k-means clustering
def kmeans_clustering(boxes, k, num_iter=100):
    # Calculate areas of boxes
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # xyxy
    #print('Box Area: ', box_areas)

    # Get indexes sorted in ascending order
    indices = np.argsort(box_areas)
    #print('Indices :', indices)  # Result: Indices : [0 3 4 1 5 2]  

    # Get the biggest box
    clusters = boxes[indices[-k:]]  # Set an initial center of clusters -> Set a beginning point of clusters-
    #print('Clusters: ', clusters)

    # Initialize the existing clusters with '0'
    prev_cluster = np.zeros_like(clusters)

    # 
    for _ in range(num_iter):
        # Calculate distances in allocation steps -> Connect the closest clusters -> Get indexces
        box_clusters = np.argmin(((boxes[:, None] - clusters[None]) ** 2).sum(axis=2),  # 'boxes[:, None] - clusters[None]': Broadcasting to make size the same
                                  axis=1)
        #print('box_clusters: ', box_clusters)

        # Update the center of clusters by calculating average values of each box
        for cluster_idx in range(k):
            if np.any(box_clusters == cluster_idx):
                clusters[cluster_idx] = boxes[box_clusters == cluster_idx].mean(axis=0)

        # Determine coverage by calcuating amount of change in clusters
        #  -> Run cluster algorithm iterations to get cluster changes -> Exit if less than threshold
        if np.all(np.abs(prev_cluster - clusters) < 1e-6):  # (Cluster changes < Threshold)
            break

        # Update values of prev_cluster
        prev_cluster = clusters.copy()
        #print('Updated center of clusters: ', prev_cluster)

    return clusters


# Def
def plot_boxes(boxes, title='Anchors'):
    fig, ax = plt.subplots(1)
    ax.set_title(title)

    # Set width and height of figure
    img_width, img_height = 200, 200

    # Display normalized axes of anchor box
    for box in boxes:
        x_min, y_min, x_max, y_max = box

        # Scale anchor boxes
        x_min, x_max = x_min / img_width, y_min / img_height
        y_min, y_max = x_max / img_width, y_max / img_height

        # Get width and height
        width, height = x_max - x_min, y_max - y_min
        rectangle = patches.Rectangle((x_min, y_min), 
                                      width, height, 
                                      linewidth = 1,
                                      edgecolor = 'r',
                                      facecolor = 'none')
        ax.add_patch(rectangle)

    # Set values 
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    plt.show()


# Create dummy boxes
boxes = np.array([[10, 10, 50, 50], [30, 20, 80, 100], [100, 90, 150, 200], 
                  [30, 30, 80, 80], [50, 55, 110, 120],[ 80, 70, 150, 130]])


# Run codes
k = 5
anchors = kmeans_clustering(boxes, k)
print('Aspect Ratio of Anchor boxes: ', anchors)
plot_boxes(anchors)