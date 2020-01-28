## Simultaneous Mapping and Target Driven Navigation [[pdf]](https://arxiv.org/pdf/1911.07980.pdf)
G. Georgakis, Y. Li, J. Kosecka, arXiv 2019

This is Python code for training and testing both mapnet and navigation components.

### Requirements
- Python 3.7
- Torch 0.4.1

#### Python Packages
```
networkx
torchvision 0.2.1
```

### Datasets

#### Active Vision Dataset (AVD)
We used the AVD dataset which can be found [here](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/). We provide additional required data (such as detections, semantic segmentations, annotated targets etc.) [here](https://cs.gmu.edu/~ggeorgak/ActiveVisionDataset.zip). The file contains a folder "ActiveVisionDataset" that also shows the expected structure of the folders when running the code. In each scene folder (i.e. "Home_001_1") put the "high_res_depth" and "jpg_rgb" folders that can be downloaded fromt the official AVD website. Finally, in parameters.py, update the avd_root path to point to the "ActiveVisionDataset" folder.
The target annotations and the chosen test set were provided by: "Mousavian et al, Visual Representation for Semantic Target Driven Navigation".

Support for Matterport3D is coming soon.


### MapNet Module
The implementation follows the module described in "Henriques et al, MapNet: An Allocentric Spatial Memory for Mapping Environments" along with our own extensions. Run the `train_MapNet.py` and `test_MapNet.py` directly for training and testing respectively. Make sure to edit the `parameters.py` under the ParametersMapNet class first. 

### Navigation Module
The implementation is tailored for imitation learning. The model definition can be found in `IL_Net.py`. Run `train_NavNet.py` and `test_NavNet.py` for training and testing after editing the appropriate parameters in the Parameters_IL class.

### Visualizations
Use the following for visualizing any results:
```shell
python visualize_episodes.py \
  --avd_root=".../ActiveVisionDataset/" \
  --episode_results_file=".../episode_results_eval_0.pkl" \
  --scene="Home_001_1" \
  --save_path=".../examples/" \
  --mode="nav"
```
The script requires installing the python `plyfile` module in order to read the point clouds in AVD provided in `.ply` format. Also make sure that each scene folder contains the appropriate `dense_reconstruction.ply` provided with the rest of the AVD dataset. The `episode_results.pkl` files can be created by running `test_NavNet.py` or `test_MapNet.py` for generating results for navigation and localization respectively.

### Citation
If you use this code for your research, please consider citing:
```
@article{georgakis2019simultaneous,
  title={Simultaneous Mapping and Target Driven Navigation},
  author={Georgakis, Georgios and Li, Yimeng and Kosecka, Jana},
  journal={arXiv preprint arXiv:1911.07980},
  year={2019}
}
```
