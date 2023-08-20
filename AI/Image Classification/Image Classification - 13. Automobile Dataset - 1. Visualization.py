# Import modules
import cv2
import matplotlib.pyplot as plt


# Define a function to draw bounding boxes
def draw_bboxes_on_images(file_images, file_annotations):
    # Load images
    image = cv2.imread(file_images)
    #print(image)

    # Read annotation files
    with open(file_annotations, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    #print(lines)  # Result: ['24.0 1099 257 1354 257 1354 549 1099 549\n', '27.0 682 178 930 178 930 449 682 449\n']
    for line in lines:
        # Get each line from annotations
        values = list(map(float, line.strip().split(' ')))
        #print(values)  # Result: [24.0, 1099.0, 257.0, 1354.0, 257.0, 1354.0, 549.0, 1099.0, 549.0]
                        #         [27.0, 682.0, 178.0, 930.0, 178.0, 930.0, 449.0, 682.0, 449.0]
        # Get class IDs
        class_id = int(values[0])

        # Get axes of bounding boxes
        #  - Reasons of 8 axis points
        #    : Use 4 points way (LabelMe method) -> 4 pairs of axes
        #      e.g. x1, y1, x2, y2, x3, y3, x4, y4
        #  - Min values: Starting point (Axis of Left & Up corner)
        #  - Max values: Ending point (Axis of Right & Down corner) -> Get the max value among 3 axes
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))
        print(class_id, x_min, y_min, x_max, y_max)

        # Draw bounding boxes
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add labels
        cv2.putText(image, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 1)

    # Visualization
    cv2.imshow('Test', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()

# Run codes
if __name__ == '__main__':
    # Set paths
    path_file_images = './datasets/0725-Automobile Dataset/train/syn_00010.png'
    path_file_annotations = './datasets/0725-Automobile Dataset/train/syn_00010.txt'

    draw_bboxes_on_images(path_file_images, path_file_annotations)