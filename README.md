The official code release of Enhancing Point Cloud Tracking with Spatio-temporal Triangle Optimization.

This is a rough implementation, we are working on it.

## Setup
Installation
+ Create the environment
  ```
  git clone https://github.com/Ghostish/Open3DSOT.git
  cd Open3DSOT
  conda create -n Open3DSOT  python=3.8
  conda activate Open3DSOT
  ```
+ Install pytorch
  ```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
  ```
  Our code is compatible with other PyTorch/CUDA versions. You can follow [this](https://pytorch.org/get-started/locally/) to install another version of pytorch. **Note: In order to reproduce the reported results with the provided checkpoints of BAT, please use CUDA 10.x.** 

+ Install other dependencies:
  ```
  pip install -r requirement.txt
  ```


KITTI dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

NuScenes dataset
+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.

```

## Quick Start
### Training
To train a model, you must specify the `.yaml` file with `--cfg` argument. The `.yaml` file contains all the configurations of the dataset and the model. We provide `.yaml` files under the [*cfgs*](./cfgs) directory. **Note:** Before running the code, you will need to edit the `.yaml` file by setting the `path` argument as the correct root of the dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --cfg cfgs/M2_track_kitti.yaml  --batch_size 64 --epoch 60 --preloading
### Testing
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.
```bash
python main.py  --cfg cfgs/M2_track_kitti.yaml  --checkpoint ./pretrained_models/mmtrack_kitti_car.ckpt --test
```


