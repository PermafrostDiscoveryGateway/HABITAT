from final_model_config import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from torch.utils.data import DataLoader
from dataloader import *
from tqdm import tqdm
import itertools

def model_evaluation():

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

    test_dataloader = DataLoader(test_dataset)

    test_dataset_vis = Dataset(
        x_test_dir,
        y_test_dir
    )

    # Evaluate model on test dataset

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=Final_Config.LOSS,
        metrics=Final_Config.METRICS,
        device=Final_Config.DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    # Create function to visualize predictions


    def visualize(**images):
        """Plot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()


    # Visualize predictions on test dataset.


    for i, id_ in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        
        image_vis = test_dataset_vis[i][0].astype('float')
        image_vis = image_vis/65535
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        predicted_mask = np.moveaxis(pr_mask, 0, 2)

        visualize(
        image=image_vis,
        ground_truth_mask=np.argmax(np.moveaxis(gt_mask, 0, 2), axis=2),
        predicted_mask=np.argmax(predicted_mask, axis=2)
        )

        name = Final_Config.TEST_OUTPUT_DIR + '/test_preds/' + str(i) + '.png'
        cv2.imwrite(name, np.argmax(predicted_mask, axis=2))


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
    labels_max = np.argmax(labels, 1)
    labels_max_f = labels_max.flatten()

    # Construct confusion matrix and calculate classification metrics with sklearn

    cm = confusion_matrix(labels_max_f, preds_max_f)
    report = classification_report(labels_max_f, preds_max_f)
    print(report)

    # Define function to plot confusion matrix 

    classes = ['Background', 'Detached house', 'Row house', 'Multi-story block', 'Non-residential building', 'Road', 'Runway', 'Gravel pad', 'Pipeline', 'Tank']


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
        plt.savefig(Final_Config.TEST_OUTPUT_DIR + '/confusion_matrix' + '.png', dpi = 1000, bbox_inches = "tight")


    # Plot confusion matrix
    plt.figure(figsize=(7, 7))
    plot_confusion_matrix(cm, classes)