# Final Data Science Project

This is the my final data science project on applying deep learning to detect blurry images.

This repo includes the following core code:

1. 1_import_packages_functions.ipynb
  * imported packages
  * import predefined functions, which will be used later in the code


2. 2-1_read_data_cat.ipynb
  * read the images file from the folder, particularly for the cat images which were used in the slides for illustration


3. 2-2_read_data_save_2_labels_clear_blurry.ipynb
  * read images from the folder
  * convert to numpy matrix
  * save the matrix as separate files (to be loaded in the CNN model)


4. 2-3_read_data_save_5_labels.ipynb
  * This is the same as the previous one, except for the datasets for more than 2 labels, when including images blurred with various algorithms and out-of-focus fix_image_size


5. 3_model_CNN.ipynb
  * import the data
  * CNN modeling with tensorflow and keras


6. 4_model_CNN_validation.ipynb
  * Validate the model with my own data, such as dog images.


*Appendix folder*: miscellaneous code for image analysis and data preprocessing


*notebook_with_results*: multiple saved pdf for the trained results
