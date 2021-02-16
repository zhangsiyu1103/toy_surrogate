
This is a toy example to examine the surrogate idea:


- To generate data:
  ```
  python generate_data.py
  ```
  This would generate 400 training data and 100 testing data on a regressor with 6k parameters. The data would be stored in ```dataset/data/```

- To train the regressor

  ```
  python train_teacher.py
  ```
  This would train the regressor and save the model to $TRAIN_MODEL_PATH. By default, it would be stored in the folder ```save```


  To use the surrogate model

  ```
  python train_teacher.py --surrogate --suffix surrogate --teacher $TRAIN_MODEL_PATH
  ```

