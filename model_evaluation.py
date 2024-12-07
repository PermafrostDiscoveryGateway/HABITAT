from final_model_config import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from torch.utils.data import DataLoader
from dataloader import *
from tqdm import tqdm
import itertools
from segmentation_models_pytorch import utils

def model_evaluation():

    out_dir = Final_Config.TEST_OUTPUT_DIR + Final_Config.NAME
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = out_dir + '/evaluation_results'
    os.makedirs(pred_dir, exist_ok=True)

    # Evaluation and Visualization

    # load best saved checkpoint

    model_path = Final_Config.WEIGHT_DIR + '.pth'
    best_model = torch.load(model_path)

    # Create test dataset for model evaluation and prediction visualization

    x_test_dir = Final_Config.INPUT_IMG_DIR + '/test'
    y_test_dir = Final_Config.INPUT_MASK_DIR + '/test'

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        preprocessing=get_preprocessing(Final_Config.PREPROCESS),
    )

    # test_dataloader = DataLoader(test_dataset)

    test_dataset_vis = Dataset(
        x_test_dir,
        y_test_dir
    )

    # Evaluate model on test dataset

    test_epoch = utils.train.ValidEpoch(
        model=best_model,
        loss=Final_Config.LOSS,
        metrics=Final_Config.METRICS,
        device=Final_Config.DEVICE,
    )

    # logs = test_epoch.run(test_dataloader)

    # Create function to visualize predictions

    # Function to overlay the predicted mask on top of the original image
    def overlay_mask(image, predicted_mask):
        # Define color mappings for each class
        color_map = {
            1: (255, 255, 0),  # Yellow for class 1
            2: (0, 0, 255),  # Blue for class 2
            3: (34, 139, 34)  # Forest Green for class 3
        }

        # Create an empty overlay with the same shape as the input image
        overlay = np.zeros_like(image, dtype=np.uint8)

        # Map mask values to their corresponding colors
        for class_value, color in color_map.items():
            overlay[predicted_mask == class_value] = color

        # Convert overlay to the same format as the input image
        overlay = overlay.astype(np.uint8)

        # Blend the original image with the overlay
        alpha = 0.5  # Transparency factor
        blended_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return blended_image

    # Update the visualization loop to use the overlay function
    for i, id_ in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        # Prepare the input image for visualization
        image_vis = (test_dataset_vis[i][0]).astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB) / 255.0

        # Prepare ground truth and predicted masks
        image, gt_mask = test_dataset[i]
        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        predicted_mask = np.moveaxis(pr_mask, 0, 2)

        # Overlay predicted mask on the input image
        blended_image = overlay_mask((image_vis * 255).astype(np.uint8), np.argmax(predicted_mask, axis=2))

        # Save the blended image
        name = pred_dir + '/' + str(i) + '.png'
        plt.imsave(name, blended_image)
        print(f"Saved image with overlay: {name}")


    # Run inference on test images and store the predictions and labels
    # in arrays to construct confusion matrix.

    # Get the number of files in the test dataset in order to create the label and prediction arrays
    files = [f for f in os.listdir(x_test_dir) if os.path.isfile(os.path.join(x_test_dir, f))]
    num_files = len(files)

    labels = np.empty([num_files, Final_Config.CLASSES, Final_Config.SIZE, Final_Config.SIZE])
    preds = np.empty([num_files, Final_Config.CLASSES, Final_Config.SIZE, Final_Config.SIZE])
    for i, id_ in tqdm(enumerate(test_dataset), total = len(test_dataset)):
        
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        labels[i] = gt_mask
        
        x_tensor = torch.from_numpy(image).to(Final_Config.DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        preds[i] = pr_mask


    # Prepare prediction and label arrays for confusion matrix by deriving the predicted class for each sample and
    # flattening the arrays

    preds_max = np.argmax(preds, 1)
    preds_max_f = preds_max.flatten()
    # preds_f = preds.flatten()
    labels_max = np.argmax(labels, 1)
    labels_max_f = labels_max.flatten()
    # labels_f = labels.flatten()

    # Construct confusion matrix and calculate classification metrics with sklearn

    cm = confusion_matrix(labels_max_f, preds_max_f)
    report = classification_report(labels_max_f, preds_max_f)
    iou_report = jaccard_score(labels_max_f, preds_max_f, average=None)
    acc_report = accuracy_score(labels_max_f, preds_max_f)
    print(iou_report)
    print(acc_report)
    print(report)

    # Define function to plot confusion matrix 

    # For full classification scheme
    # classes = ['Background', 'Detached house', 'Row house', 'Multi-story block', 'Non-residential building', 'Road', 'Runway', 'Gravel pad', 'Pipeline', 'Tank']

    # # For classification scheme with single building class
    # classes = ['Background', 'Building', 'Road', 'Runway', 'Pipeline', 'Tank']

    # For classification scheme with buildings and roads only
    classes = ['Background', 'Building', 'Road', 'Tank']

    # # For classification scheme with roads only
    # classes = ['Background', 'Road']

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(out_dir + '/confusion_matrix' + '.png', dpi = 1000, bbox_inches = "tight")


    # Plot confusion matrix

    # For full classification scheme
    # plt.figure(figsize=(10, 10))

    # For classification scheme with two classes
    plt.figure(figsize=(3, 3))
    plot_confusion_matrix(cm, classes)